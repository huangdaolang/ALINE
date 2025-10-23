import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import os
import random
import numpy as np
import time

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from model import Aline
from utils import create_logger, set_seed, save_state_dict, compute_ll, load_checkpoint, save_checkpoint, create_target_mask, select_targets_by_mask, set_layerwise_lr, eval_boed


os.environ["HYDRA_FULL_ERROR"] = "1"

def train(cfg, logger, model, experiment, batch_size: int, min_T: int, max_T: int, max_epoch: int, verbose: int = 10):
    """ Mutlitask Learning

    Args:
        batch_size (int): number of parallel experiments
        min_T (int): minimum value of T
        max_T (int): maximum value of T
        max_epoch (int): max epoch of optimisation
        verbose (int): interval to show the process.
    """
    # Optimizer
    optimizer, scheduler = set_layerwise_lr(cfg, model)

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    # Load from checkpoint if specified
    if cfg.load_checkpoint is True:
        start_epoch, optimizer, scheduler = load_checkpoint(cfg, model, optimizer, scheduler, cfg.load_path)
    else:
        start_epoch = 0

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    # Set smaller query size during burining
    if start_epoch < cfg.burning_epoch:
        experiment.n_query_init = cfg.T

    losses = []
    training_times = []

    for epoch in range(start_epoch, max_epoch):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()

        # Data
        T = random.randint(min_T, max_T)
        batch = experiment.sample_batch(batch_size)

        mask_type = random.choice(cfg.task.mask_type)
        batch.target_mask = create_target_mask(mask_type,
                                               cfg.task.embedding_type,
                                               cfg.task.n_target_data,
                                               cfg.task.n_target_theta,
                                               cfg.task.n_selected_targets,
                                               cfg.task.predefined_masks,
                                               cfg.task.predefined_mask_weights,
                                               cfg.task.mask_index,
                                               cfg.task.attend_to,
                                               )

        # log probs of design history
        log_probs = torch.zeros((batch_size, T))  # [B, T]
        nlls_for_prediction = []
        nlls_for_query = []

        # T-steps experiment
        for t in range(T):
            if cfg.time_token:
                batch.t = torch.tensor([t/T])

            pred = model.forward(batch)  # idx: [B, 1], log_prob: [B], zt: [B, n_query]
            design_out = pred.design_out
            posterior_out = pred.posterior_out

            batch = experiment.update_batch(batch, design_out.idx)

            log_probs[:, t] = design_out.log_prob                # [B]

            target_ll = compute_ll(batch.target_all,
                                   posterior_out.mixture_means,
                                   posterior_out.mixture_stds,
                                   posterior_out.mixture_weights)  # [B, n_target]

            masked_target_ll = select_targets_by_mask(target_ll, batch.target_mask)

            if cfg.task.embedding_type == "mix" and mask_type == "all":
                nll_for_query = - (masked_target_ll[:, :-cfg.task.n_target_theta].mean(dim=-1) +
                                   masked_target_ll[:, -cfg.task.n_target_theta:].mean(dim=-1))
            else:
                nll_for_query = - masked_target_ll.mean(dim=-1)
            nlls_for_query.append(nll_for_query)

            if cfg.task.embedding_type == "mix":
                nll = - (target_ll[:, :-cfg.task.n_target_theta].mean(dim=-1) + target_ll[:, -cfg.task.n_target_theta:].mean(dim=-1))
            else:
                nll = - target_ll.mean(dim=-1)
            nlls_for_prediction.append(nll)

        # accumulated rewards
        R = []

        for t in range(1, T):
            # likelihood gain
            ll_gain = torch.clamp(nlls_for_query[t - 1] - nlls_for_query[t], min=0.0).detach()  # [B]
            reward = (cfg.gamma ** t) * ll_gain
            R.append(reward)

        R = torch.stack(R, 1)                                               # [B, T-1]
        R = (R - R.mean(dim=0, keepdim=True)) / (R.std(dim=0, keepdim=True) + 1e-9)  # normalize

        design_loss = - torch.mean(log_probs[:, :-1] * R)        # xi_t * (nll_t - nll_{t+1})
        predict_loss = torch.mean(torch.stack(nlls_for_prediction))
        if epoch < cfg.burning_epoch:
            loss = predict_loss
            loss.backward()

        else:
            loss = design_loss * cfg.alpha + predict_loss
            loss.backward()

        losses.append(loss.item())

        # Gradient clipping
        if cfg.clip_grads:
            clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type="inf")


        # Setting different learning rates for shared layers after the burning stage
        if epoch == cfg.burning_epoch:
            optimizer, scheduler = set_layerwise_lr(cfg, model, epoch)

            # reset query size
            experiment.n_query_init = cfg.task.n_query_init

            # save the model
            logger.info(f"""Model has been saved at {save_state_dict(model, cfg.output_dir, f'{cfg.file_name.split(".")[0]}_burning.pth')}""")

        optimizer.step()
        scheduler.step()

        end_time = time.time()
        training_times.append(end_time - start_time)

        if cfg.wandb.use_wandb:
            wandb.log({"loss": loss, "likelihood": -predict_loss, "design_loss": design_loss, "targeted_likelihood": -torch.mean(torch.stack(nlls_for_query))}, step=epoch)

        if epoch % verbose == 0:
            logger.info(f"Epoch: {epoch}, loss: {losses[-1]:.4f}, T: {T}, likelihood: {-predict_loss}, design_loss: {design_loss}, predict_loss: {predict_loss}")

            if cfg.eval.EIG:
                bounds = eval_boed(model, experiment, cfg.T - cfg.task.n_context_init, cfg.eval.L, cfg.eval.M, cfg.eval.batch_size, cfg.time_token, False)
                pce_loss = bounds['pce_mean']
                nmc_loss = bounds['nmc_mean']
                logger.info(f"PCE: {pce_loss}\tNMC: {nmc_loss}")
                if cfg.wandb.use_wandb:
                    wandb.log({"PCE": pce_loss, "NMC": nmc_loss}, step=epoch)

        next_epoch = epoch + 1
        if cfg.checkpoint and next_epoch % cfg.checkpoint == 0:
            save_checkpoint(cfg, model, optimizer, scheduler, next_epoch, with_epoch=True)

    total_time = sum(training_times)
    average_time = np.mean(training_times[cfg.burning_epoch:])
    std_time = np.std(training_times[cfg.burning_epoch:])
    logger.info(f"Total training time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours), average time per epoch: {average_time:.2f}+-{std_time:.2f} seconds")

    if cfg.wandb.use_wandb:
        wandb.log({"training_time": total_time}, step=max_epoch)


@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg):
    # Setting device
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    torch.set_default_device(cfg.device)
    if cfg.device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Setting random seed
    if cfg.fix_seed:
        set_seed(cfg.seed)
    else:
        cfg.seed = torch.random.seed()

    cfg.output_dir = str(HydraConfig.get().runtime.output_dir)

    # Ensure min_T is not larger than T
    if cfg.min_T > cfg.T:
        cfg.min_T = cfg.T
    
    # Create logger
    logger = create_logger(os.path.join(cfg.output_dir, 'logs'), name=cfg.task.name)
    logger.info("Running with config:\n{}".format(OmegaConf.to_yaml(cfg)))

    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.output_dir,
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            hydra_log_dir = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    # Data
    experiment = hydra.utils.instantiate(cfg.task)
    logger.info(f"Task: {experiment}")

    # For HPO tasks, validate that we have an HPO task and update config dimensions
    if hasattr(experiment, 'meta_dataset'):
        logger.info(f"Using HPO-B meta-dataset: {experiment.meta_dataset}")
        logger.info(f"Input dimension: {experiment.dim_x}")
        logger.info(f"Number of datasets: {experiment.hpob.n_dataset}")

        # Update config with actual dimensions from the dataset
        if cfg.task.dim_x != experiment.dim_x:
            logger.info(
                f"Updating dim_x from config value {cfg.task.dim_x} to actual dataset dimension {experiment.dim_x}")
            cfg.task.dim_x = experiment.dim_x

        if cfg.task.dim_y != experiment.dim_y:
            logger.info(
                f"Updating dim_y from config value {cfg.task.dim_y} to actual dataset dimension {experiment.dim_y}")
            cfg.task.dim_y = experiment.dim_y

    # Model
    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)
    model = Aline(embedder, encoder, head)
    logger.info(model)

    if cfg.wandb.use_wandb:
        wandb.watch(model, log_freq=10)

    # Train
    train(cfg, logger, model, experiment, cfg.batch_size, cfg.min_T, cfg.T, cfg.max_epoch, verbose=cfg.verbose)

    # Save
    logger.info(f"Model has been saved at {save_state_dict(model, cfg.output_dir, cfg.file_name)}")

    # Eval
    if cfg.eval.EIG:
        # Set a larger query size
        experiment.n_query_init = cfg.eval.n_query_final

        bounds = eval_boed(model, experiment, cfg.eval.T_final - cfg.task.n_context_init, cfg.eval.L_final, cfg.eval.M_final, cfg.eval.batch_size_final, cfg.time_token, stepwise=True)

        logger.info(bounds)
        logger.info(f"PCE: {bounds['pce_mean'][cfg.T-1]:.3f}+-{bounds['pce_err'][cfg.T-1]:.3f}\tNMC: {bounds['nmc_mean'][cfg.T-1]:.3f}+-{bounds['nmc_err'][cfg.T-1]:.3f}")

        # save bounds to file
        save_path = os.path.join(cfg.output_dir, "eval", f"{cfg.file_name.split('.')[0]}_N{cfg.eval.n_query_final}_T{cfg.eval.T_final}.tar")
        # make dir if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(bounds, save_path)
        logger.info(f"Bounds have been saved at {save_path}.")
                           




if __name__ == '__main__':
    main()
