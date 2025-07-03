import math
import torch
import torch.nn as nn
import torch.distributions as dist
    

class EIGBounds(nn.Module):
    def __init__(self, L: int, T: int, log_prob, reduction=None) -> None:
        """ Base class for EIG Bounds, computed at the end of trajectories

        Args:
            L (int): number of contrastive samples $\theta_{1:L}$
            T (int): number of designs in each trajectory
            log_prob (function): function to calculate the log likelihoods given theta and design policy
        """
        super(EIGBounds, self).__init__()
        self.L = L
        self.T = T
        self.log_prob = log_prob
        self.reduction = reduction

    def compute_seq_logprobs(self, y_outcomes, xi_designs, thetas):
        """ Compute the sequential log likelihood

        Args:
            y_outcomes [B, T, 1]: a batch of sequential experiment's outcomes
            xi_designs [B, T, D]: a batch of designs
            thetas [L, B, K, D] or [L, B, D]: L batches of latent variables sampled from the prior

        Returns:
            log_probs [L, B]
        """
        B, T, D = xi_designs.shape
        # L, _, K, _ = thetas.shape

        # align dimensions for broadcasting
        if len(thetas.shape) == 3:
            thetas = thetas.unsqueeze(2).expand(-1, -1, T, -1)                   # [L, B, T, D]
        else:
            thetas = thetas.unsqueeze(2).expand(-1, -1, T, -1, -1)               # [L, B, T, K, D]
        xi_designs = xi_designs.unsqueeze(0)#.expand(L, -1, -1, -1)              # [1, B, T, D]
        y_outcomes = y_outcomes.unsqueeze(0)#.expand(L, -1, -1, -1)              # [1, B, T, 1]

        log_probs = self.log_prob(y_outcomes, xi_designs, thetas)              # [L, B, T, 1]

        # compute the sequential joint log likelihood
        log_probs = log_probs.sum(dim=(-2, -1))                                       # [L, B]
        return log_probs

    def forward(self, y_outcomes, xi_designs, thetas):
        # raise NotImplementedError
        return self.compute_seq_logprobs(y_outcomes, xi_designs, thetas)
    

class PCELoss(EIGBounds):
    def __init__(self, L: int, T: int, log_prob, reduction='mean') -> None:
        r""" sPCE loss (negative value of sPCE bound with the constant term removed)
            \mathbb{E}_{p ( \theta_{0}, h_{T} | \pi) p ( \theta_{1:L} )} 
            [ log { \sum_{\ell=0}^{L} p ( h_{T} | \theta_{\ell}, \pi)} - log {p ( h_{T} | \theta_{0}, \pi)}]

        Args:
            L (int): number of contrastive samples $\theta_{1:L}$
            T (int): number of designs in each trajectory
            log_prob (function): function to calculate the log likelihoods given theta and design policy
        """
        super(PCELoss, self).__init__(L, T, log_prob, reduction)

    def forward(self, y_outcomes, xi_designs, thetas):
        """ Compute the sPCE loss

        Args:
            y_outcomes [B, T, 1]: a batch of sequential experiment's outcomes
            xi_designs [B, T, D]: a batch of designs
            thetas [L, B, K, D] or [L, B, D]: L batches of latent variables sampled from the prior

        Returns:
            sPCE loss
        """
        # compute the sequential joint log likelihood
        log_probs = self.compute_seq_logprobs(y_outcomes, xi_designs, thetas)   # [L, B]
        # inner expectation
        loss = log_probs.logsumexp(0) - log_probs[0]                            # [B]
        # outer expectation
        if self.reduction == "mean":
            loss = torch.mean(loss)
        return loss
    
class PCELossScoreGradient(PCELoss):
    def __init__(self, L: int, T: int, log_prob, reduction='mean') -> None:
        super(PCELossScoreGradient, self).__init__(L, T, log_prob, reduction)

    def forward(self, y_outcomes, xi_designs, thetas):
        """ Compute the sPCE score gradient loss
        Args:
            y_outcomes [B, T, 1]: a batch of sequential experiment's outcomes
            xi_designs [B, T, D]: a batch of designs
            thetas [L, B, K, D] or [L, B, D]: L batches of latent variables sampled from the prior

        Returns:
            sPCE score gradient loss
        """

        # compute the sequential joint log likelihood
        log_probs = self.compute_seq_logprobs(y_outcomes, xi_designs, thetas)   # [L, B]

        log_prob_primary = log_probs[0]
        log_probs = log_probs.logsumexp(0)
        with torch.no_grad():
            g_no_grad = log_prob_primary - log_probs

        loss = - (g_no_grad * log_prob_primary - log_probs) # [B]
        
        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss
    

class NMCLoss(EIGBounds):
    def __init__(self, L: int, T: int, log_prob, reduction='mean') -> None:
        r""" sNMC loss (negative value of sNMC bound with the constant term removed)
            \mathbb{E}_{p ( \theta_{0}, h_{T} | \pi) p ( \theta_{1:L} )} 
            [ log { \sum_{\ell=1}^{L} p ( h_{T} | \theta_{\ell}, \pi)} - log {p ( h_{T} | \theta_{0}, \pi)}]

        Args:
            L (int): number of contrastive samples $\theta_{1:L}$
            T (int): number of designs in each trajectory
            log_prob (function): function to calculate the log likelihoods given theta and design policy
        """
        super(NMCLoss, self).__init__(L, T, log_prob, reduction)

    def forward(self, y_outcomes, xi_designs, thetas):
        """ Compute the sNMC loss

        Args:
            y_outcomes [B, T, 1]: a batch of sequential experiment's outcomes
            xi_designs [B, T, D]: a batch of designs
            thetas [L, B, K, D] or [L, B, D]: L batches of latent variables sampled from the prior

        Returns:
            sNMC loss
        """
        # compute the sequential joint log likelihood
        log_probs = self.compute_seq_logprobs(y_outcomes, xi_designs, thetas)   # [L, B]
        # inner expectation
        loss = log_probs[1:].logsumexp(0) - log_probs[0]                        # [B]
        # outer expectation
        if self.reduction == "mean":
            loss = torch.mean(loss)
        return loss 
    

class EIGStepLoss(nn.Module):
    def __init__(self, L: int, M: int, log_prob, reduction=None) -> None:
        """ Base class for EIG Bounds, compute at each step

        Args:
            L (int): number of contrastive samples $\theta_{1:L}$
            M (int): number of parallel trajectories, i.e. batch_size
            log_prob (function): function to calculate the log likelihoods given theta and design policy
        """
        super(EIGStepLoss, self).__init__()
        self.L = L
        self.M = M
        self.log_prob = log_prob
        self.reduction = reduction
        self.seq_logprobs = torch.zeros((L+ 1, M))

    def reset(self):
        """ Reset the sequential log likelihood """
        self.seq_logprobs = torch.zeros((self.L + 1, self.M))

    def step(self, y_outcomes, xi_designs, thetas):
        """ Accumulate the sequential log likelihood

        Args:
            y_outcomes [M, D_y]: a batch of sequential experiment's outcomes
            xi_designs [M, D_x]: a batch of designs
            thetas [L, M, (K, )D]: L batches of latent variables sampled from the prior

        Returns:
            log_probs [L, M]
        """

        xi_designs = xi_designs.unsqueeze(0)              # [1, B, D_x]
        y_outcomes = y_outcomes.unsqueeze(0)              # [1, B, D_y]

        log_probs = self.log_prob(y_outcomes, xi_designs, thetas).squeeze(-1)              # [L, B]

        # accumulate the sequential joint log likelihood
        self.seq_logprobs += log_probs      # [L, M]
        return self.seq_logprobs

    def forward(self, y_outcomes, xi_designs, thetas):
        log_probs = self.step(y_outcomes, xi_designs, thetas)

        # Inner expectation
        # lower bound
        pce_loss = log_probs.logsumexp(0) - log_probs[0]                            # [M]
        # upper bound
        nmc_loss = log_probs[1:].logsumexp(0) - log_probs[0]                            # [M]

        # Outer expectation
        if self.reduction == "mean":
            pce_loss = torch.mean(pce_loss)
            nmc_loss = torch.mean(nmc_loss)

        return pce_loss, nmc_loss