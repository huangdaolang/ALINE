import torch
from torch.distributions import Normal
import numpy as np
from loss.eig import EIGStepLoss
from attrdictionary import AttrDict


@torch.no_grad()
def get_traces(model, experiment, T=30, batch_size=40, time_token=False):
    """ Get history

    Args:
        T (int): number of proposed designs in a trajectory
        batch_size (int): number of rollouts
        time_token (bool): whether to use time token
    """
    model.eval()

    theta_shape = experiment.sample_theta((batch_size)).shape

    # sample a batch of normalised data
    batch = experiment.sample_batch(batch_size)

    for t in range(T):
        if time_token:
            batch.t = torch.tensor([(T-t)/T])

        idx = model.forward(batch).design_out.idx      # [B, 1]

        batch = experiment.update_batch(batch, idx)

    # collect theta0
    theta_0 = batch.target_theta.reshape(*theta_shape)      # [B, (K, )D]

    # collect history
    x = experiment.unnormalise_design(batch.context_x)      # [B, T, D_x]
    y = batch.context_y                                     # [B, T, D_y]

    return theta_0, x, y


@torch.no_grad()
def compute_EIG_from_history(experiment, theta_0, x, y, L=int(1e6), batch_size=40, stepwise=False):
    """ Evaluate the lower and upper bounds of EIG from a minibatch of the history

    Args:
        theta_0 (torch.Tensor) [B, (K, )D]: initial theta
        x (torch.Tensor) [B, T, D_x]: history of designs
        y (torch.Tensor) [B, T, D_y]: history of outcomes
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        batch_size (int): mini batch size of outer samples
    """
    T = x.shape[1]

    criterion = EIGStepLoss(L, batch_size, experiment.log_likelihood, reduction='none')

    pce_losses = []
    nmc_losses = []

    thetas = experiment.sample_theta((L, batch_size))
    thetas = torch.concat([theta_0.unsqueeze(0), thetas], dim=0)          # [L+1, B, (K, )D]

    if stepwise:
        for t in range(T):
            pce_loss, nmc_loss = criterion(y[:, t], x[:, t], thetas)  # [B]
            pce_losses.append(pce_loss)
            nmc_losses.append(nmc_loss)
                
        pce_losses = torch.stack(pce_losses, dim=-1)  # [B, T]
        nmc_losses = torch.stack(nmc_losses, dim=-1)  # [B, T]
    else:
        for t in range(T):
            pce_losses, nmc_losses = criterion(y[:, t], x[:, t], thetas)  # [B]

    # Calculate bounds
    pce_losses = torch.log(torch.tensor(L + 1)) - pce_losses  # [B(, T)]
    nmc_losses = torch.log(torch.tensor(L)) - nmc_losses      # [B(, T)]  

    return pce_losses, nmc_losses


@torch.no_grad()
def eval_EIG_from_history(experiment, theta_0, x, y, L=int(1e6), M=2000, batch_size=40, stepwise=False, err_type='se'):
    """ Evaluate the lower and upper bounds of EIG from the history

    Args:
        theta_0 (torch.Tensor) [B, (K, )D]: initial theta
        x (torch.Tensor) [B, T, D_x]: history of designs
        y (torch.Tensor) [B, T, D_y]: history of outcomes
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        batch_size (int): mini batch size of outer samples
    """
    T = x.shape[1]

    max_step = (M + batch_size - 1) // batch_size

    pce_list = []
    nmc_list = []

    for step in range(max_step):
        start_idx = step * batch_size
        end_idx = min((step + 1) * batch_size, M)
        pce_loss, nmc_loss = compute_EIG_from_history(experiment, theta_0, x[start_idx:end_idx], y[start_idx:end_idx], L, end_idx - start_idx, stepwise)
        
        pce_list.append(pce_loss)
        nmc_list.append(nmc_loss)

    # Stack bounds
    pce = torch.cat(pce_list, dim=0)   # [M(, T)]
    nmc = torch.cat(nmc_list, dim=0)   # [M(, T)]

    # Calculate mean and std
    M = pce.shape[0]
    pce_mean = torch.mean(pce, dim=0)    # [T]
    pce_err = torch.std(pce, dim=0)     # [T]
    nmc_mean = torch.mean(nmc, dim=0)    # [T]
    nmc_err = torch.std(nmc, dim=0) / np.sqrt(M)     # [T]

    # Error type
    if err_type == 'se':
        pce_err = pce_err / np.sqrt(M)
        nmc_err = nmc_err / np.sqrt(M)
    elif err_type == 'ci':
        pce_err = 1.96 * pce_err / np.sqrt(M)
        nmc_err = 1.96 * nmc_err / np.sqrt(M)
    elif err_type == 'std':
        pass
    else:
        raise ValueError(f"Unknown err_type: {err_type}")

    pce_mean = pce_mean.cpu()
    pce_err = pce_err.cpu()
    nmc_mean = nmc_mean.cpu()
    nmc_err = nmc_err.cpu()

    bounds = AttrDict(pce_mean=pce_mean, pce_err=pce_err, nmc_mean=nmc_mean, nmc_err=nmc_err)

    return bounds

@torch.no_grad()
def eval_boed(model, experiment, T=30, L=int(1e6), M=2000, batch_size=40, time_token=False, stepwise=False, err_type='se'):
    """ Final evaluation of the EIG bounds for ALINE

    Args:
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        M (int): number of outer samples
        batch_size (int): mini batch size of outer samples
        time_token (bool): whether to use time token
    """
    model.eval()

    max_step = (M + batch_size - 1) // batch_size

    pce_list = []
    nmc_list = []

    for step in range(max_step):
        theta_0, x, y = get_traces(model, experiment, T, batch_size, time_token)
        pce, nmc = compute_EIG_from_history(experiment, theta_0, x, y, L, batch_size, stepwise)
        pce_list.append(pce)
        nmc_list.append(nmc)
        print(f"Step {step}: PCE {pce.mean(dim=0)}, NMC {nmc.mean(dim=0)}")

    # Stack bounds
    pce = torch.cat(pce_list, dim=0)   # [M(, T)]
    nmc = torch.cat(nmc_list, dim=0)   # [M(, T)]

    # Calculate mean and std
    M = pce.shape[0]
    pce_mean = torch.mean(pce, dim=0)    # [T]
    pce_err = torch.std(pce, dim=0)     # [T]
    nmc_mean = torch.mean(nmc, dim=0)    # [T]
    nmc_err = torch.std(nmc, dim=0)     # [T]

    # Error type
    if err_type == 'se':
        pce_err = pce_err / np.sqrt(M)
        nmc_err = nmc_err / np.sqrt(M)
    elif err_type == 'ci':
        pce_err = 1.96 * pce_err / np.sqrt(M)
        nmc_err = 1.96 * nmc_err / np.sqrt(M)
    elif err_type == 'std':
        pass
    else:
        raise ValueError(f"Unknown err_type: {err_type}")


    pce_mean = pce_mean.cpu()
    pce_err = pce_err.cpu()
    nmc_mean = nmc_mean.cpu()
    nmc_err = nmc_err.cpu()

    bounds = AttrDict(pce_mean=pce_mean, pce_err=pce_err, nmc_mean=nmc_mean, nmc_err=nmc_err)

    return bounds

def compute_ll(value: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Computes log-likelihood loss for a Gaussian mixture model.
    """
    components = Normal(means, stds, validate_args=False)
    log_probs = components.log_prob(value)
    weighted_log_probs = log_probs + torch.log(weights)
    return torch.logsumexp(weighted_log_probs, dim=-1)


def compute_rmse(target_values, mixture_means, mixture_stds, mixture_weights):
    # TODO: you have to apply the target mask to the target_values and mixture components first, need to update later.
    """
    Compute RMSE between target values and predictions from Gaussian mixture model

    Args:
        target_values: Ground truth values [batch_size, n_targets, dim_y]
        mixture_means: Means of mixture components [batch_size, n_targets, n_components]
        mixture_stds: Standard deviations of mixture components [batch_size, n_targets, n_components]
        mixture_weights: Weights of mixture components [batch_size, n_targets, n_components]

    Returns:
        rmse: RMSE values [batch_size, n_targets]
    """
    # Calculate weighted mean for each target point
    weighted_means = torch.sum(mixture_weights * mixture_means, dim=-1)  # [batch_size, n_targets]

    # Calculate squared error
    squared_error = (target_values.squeeze(-1) - weighted_means) ** 2

    # Calculate RMSE
    rmse = torch.sqrt(torch.mean(squared_error, dim=-1))  # [batch_size]

    return rmse