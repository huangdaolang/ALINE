import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import os
import random

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from tasks import PsychometricTask
from model import BaseTransformer
from utils import create_logger, set_seed, save_state_dict, compute_ll, remove_design, load_checkpoint, save_checkpoint, add_design, create_target_mask, select_targets_by_mask


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
    optimizer = getattr(optim, cfg.optimizer)(model.parameters(), lr=cfg.lr)
    if cfg.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **cfg.scheduler_args)
    elif cfg.scheduler == "BurningWarmup":
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            [   
                lr_scheduler.LambdaLR(optimizer, lambda epoch: 10 - (9 * epoch / cfg.burning_epoch)),
                lr_scheduler.CosineAnnealingLR(optimizer, max_epoch - cfg.burning_epoch),
            ],
            milestones=[cfg.burning_epoch]
        )
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    # Load from checkpoint if specified
    if cfg.load_checkpoint == True:
        start_epoch = load_checkpoint(cfg, model, optimizer, scheduler)
    else:
        start_epoch = 0

    losses = []

    for epoch in range(start_epoch, max_epoch):
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
        log_probs = torch.zeros((batch_size, T))                            # [B, T]
        nlls_for_prediction = []
        nlls_for_query = []

        # T-steps experiment
        for t in range(T):
            if cfg.time_token:
                batch.t = torch.tensor([(T-t)/T])

            pred = model.forward(batch)         # xi: [B, D], idx: [B, 1], log_prob: [B]
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
                nll = - (target_ll[:, :-cfg.task.n_target_theta].mean(dim=-1) + target_ll[:,
                                                                                -cfg.task.n_target_theta:].mean(dim=-1))
            else:
                nll = - target_ll.mean(dim=-1)
            nlls_for_prediction.append(nll)

        # Compute loss
        # accumulated rewards
        reward = torch.zeros(batch_size)                     # [B]
        R = []

        # for t in range(T - 1, 0, -1):
        #     # likelihood gain
        #     ll_gain = torch.clamp(nlls_for_query[t - 1] - nlls_for_query[t], min=0.0).detach()
        #     reward = ll_gain + cfg.gamma * reward
        #     R.insert(0, reward.clone())

        for t in range(1, T):
            # likelihood gain
            ll_gain = torch.clamp(nlls_for_query[t - 1] - nlls_for_query[t], min=0.0).detach()  # [B]

            reward = (cfg.gamma ** t) * ll_gain
            R.append(reward)

        R = torch.stack(R, 1)  # [B, T-1]
        R = (R - R.mean(dim=0, keepdim=True)) / (R.std(dim=0, keepdim=True) + 1e-9)  # normalize

        design_loss = - torch.mean(log_probs[:, :-1] * R)        # xi_t * (nll_t - nll_{t+1})
        # predict_loss = torch.mean(nlls_for_prediction[-1])
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
            shared_params, predictor_params = [], []
            for name, param in model.named_parameters():
                if 'predictor' not in name:
                    shared_params.append(param)
                else:
                    predictor_params.append(param)

            optimizer = getattr(optim, cfg.optimizer)(
                [
                    {"params": shared_params, "lr": cfg.lr / 5},
                    {"params": predictor_params},
                ],
                lr=cfg.lr,
            )
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epoch - cfg.burning_epoch
            )

        optimizer.step()
        scheduler.step()

        if cfg.wandb.use_wandb:
            wandb.log({"loss": loss, "likelihood": -predict_loss, "design_loss": design_loss, "targeted_likelihood": -torch.mean(torch.stack(nlls_for_query))}, step=epoch)

        if epoch % verbose == 0:
            logger.info(f"Epoch: {epoch}, loss: {losses[-1]:.4f}, T: {T}, likelihood: {-predict_loss}, design_loss: {design_loss}, predict_loss: {predict_loss}")

        if cfg.checkpoint and epoch % cfg.checkpoint == 0:
            save_checkpoint(cfg, model, optimizer, scheduler, epoch + 1)


@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg):
    logger = create_logger(os.path.join(cfg.output_dir, 'logs'), name='psychometric')

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
    train(cfg, logger, model, experiment, cfg.batch_size, cfg.min_T, cfg.T, cfg.max_epoch, verbose=cfg.verbose)

    # Save
    logger.info(f"Model has been saved at {save_state_dict(model, cfg.output_dir, cfg.file_name)}")



if __name__ == '__main__':
    main()
