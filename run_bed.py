import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import os
import random
import math
import numpy as np
import time


import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf



from model import BaseTransformer
from utils.eval import eval_boed
from utils import create_logger, set_seed, save_state_dict, compute_ll, load_checkpoint, save_checkpoint
from utils.misc import set_layerwise_lr


def train(cfg, logger, model, experiment, batch_size: int, T: int, max_epoch: int, verbose: int = 10):
    """ Mutlitask Learning

    Args:
        batch_size (int): number of parallel experiments
        min_T (int): minimum value of T
        max_T (int): maximum value of T
        max_epoch (int): max epoch of optimisation
        verbose (int): interval to show the process.
    """
    optimizer, scheduler = set_layerwise_lr(cfg, model)

    # Load from checkpoint if specified
    if cfg.load_checkpoint is True:
        start_epoch, optimizer, scheduler = load_checkpoint(cfg, model, optimizer, scheduler, cfg.load_path)
    else:
        start_epoch = 0

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    losses = []

    training_times = []

    for epoch in range(start_epoch, max_epoch):
        start_time = time.time()

        model.train()
        optimizer.zero_grad()

        # Data        
        batch = experiment.sample_batch(batch_size)

        # log probs of design history
        log_probs = torch.zeros((batch_size, T))                            # [B, T]
        nlls = []

        # T-steps experiment
        for t in range(T):
            outs = model.forward(batch)  # idx: [B, 1], log_prob: [B], zt: [B, n_query]
            design_out = outs.design_out
            posterior_out = outs.posterior_out

            batch = experiment.update_batch(batch, design_out.idx)

            log_probs[:, t] = design_out.log_prob                # [B]

            target_ll = compute_ll(batch.target_all,
                                   posterior_out.mixture_means,
                                   posterior_out.mixture_stds,
                                   posterior_out.mixture_weights)  # [B, n_target]

            nll = - target_ll.mean(dim=-1)
            nlls.append(nll)
        # Compute loss
        # accumulated rewards
        reward = torch.zeros(batch_size)                     # [B]
        R = []

        for t in range(1, T):
            # likelihood gain
            ll_gain = torch.clamp(nlls[t - 1] - nlls[t], min=0.0).detach()
            reward = ll_gain * (cfg.gamma ** t)
            R.append(reward)

        R = torch.stack(R, 1)                                               # [B, T-1]
        R = (R - R.mean(dim=0, keepdim=True)) / (R.std(dim=0, keepdim=True) + 1e-9)

        design_loss = - torch.mean(log_probs[:, :-1] * R)        # xi_t * (nll_t - nll_{t+1})
        predict_loss = torch.mean(torch.stack(nlls))

        if epoch < cfg.burning_epoch:
            loss = predict_loss
            loss.backward()

        else:
            loss = design_loss * cfg.alpha + predict_loss * cfg.beta
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
            wandb.log({"loss": loss, "likelihood": -predict_loss, "design_loss": design_loss}, step=epoch)

        if epoch % verbose == 0:
            logger.info(f"Epoch: {epoch}, loss: {losses[-1]:.4f}, T: {T}, likelihood: {-predict_loss}, design_loss: {design_loss}, predict_loss: {predict_loss}")
            bounds = eval_boed(model, experiment, cfg.T - cfg.task.n_context_init, cfg.L, cfg.M, cfg.eval_batch_size, cfg.time_token, False)
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



@hydra.main(version_base=None, config_path="./config", config_name="train_bed")
def main(cfg):
    logger = create_logger(os.path.join(cfg.output_dir, 'logs'), name=cfg.task.name)

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
    logger.info(experiment)

    # Model
    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)
    model = BaseTransformer(embedder, encoder, head)
    logger.info(model)

    if cfg.wandb.use_wandb:
        wandb.watch(model, log_freq=10)


    # Train
    train(cfg, logger, model, experiment, cfg.batch_size, cfg.T, cfg.max_epoch, verbose=cfg.verbose)

    # Save
    logger.info(f"Model has been saved at {save_state_dict(model, cfg.output_dir, cfg.file_name)}")

    # Eval
    # Set a larger query size during evaluation
    experiment.n_query_init = cfg.n_query_final

    bounds = eval_boed(model, experiment, cfg.T_final - cfg.task.n_context_init, cfg.L_final, cfg.M_final, cfg.eval_batch_size_final, cfg.time_token, stepwise=True)

    logger.info(bounds)
    logger.info(f"PCE: {bounds['pce_mean'][cfg.T-1]:.3f}+-{bounds['pce_se'][cfg.T-1]:.3f}\tNMC: {bounds['nmc_mean'][cfg.T-1]:.3f}+-{bounds['nmc_se'][cfg.T-1]:.3f}")

    # save bounds to file
    save_path = os.path.join(cfg.output_dir, "eval", f"{cfg.file_name.split('.')[0]}_N{cfg.n_query_final}_T{cfg.T_final}.tar")
    # make dir if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(bounds, save_path)
    logger.info(f"Bounds have been saved at {save_path}.")


if __name__ == '__main__':
    main()
