import torch
import torch.nn as nn
import torch.distributions as dist

class NMLELoss(nn.Module):
    """ Negative values of log likelihood  \sum_{i=1}^{B}\log p(\theta_i|\mu_i^*, \sigma_i^*) """
    def __init__(self, post_dist: dist.Distribution, reduction='mean') -> None:
        super(NMLELoss, self).__init__()

        self.post_dist = post_dist
        self.reduction = reduction

    def reduce(self, x):
        if self.reduction == "mean":
            x = x.mean()
        elif self.reduction == "sum":
            x = x.sum()
        elif self.reduction == "logsumexp":
            x = torch.logsumexp(x, 0)

        return x
    

    def forward(self, output, target):
        """ Compute the likelihood 

        Args:
            output (AttriDict): prediction
            target [B, (K, D)]: thetas
        """
        log_probs = self.post_dist(**output).log_prob(target)                # [B, (K, D)]
        log_probs = log_probs.sum(dim=tuple(range(1, log_probs.dim())))      # [B]                                                            
        log_probs = self.reduce(log_probs)

        return - log_probs


class SortedNMLELoss(NMLELoss):
    """ Negative values of log likelihood  \sum_{i=1}^{B}\log p(\theta_i|\mu_i^*, \sigma_i^*) 
        with sorted targets
    """
    def __init__(self, post_dist: dist.Distribution, reduction='mean') -> None:
        super(SortedNMLELoss, self).__init__()
    

    def forward(self, output, target):
        """ Compute the likelihood 

        Args:
            output [B, K, D, 2]: prediction
            target [B, K, D]: thetas
        """
        # Sort
        target, _ = torch.sort(target, dim=1)
        output.loc, idx = torch.sort(output.loc, dim=1)
        output.scale = output.scale[:, idx]
        log_probs = self.post_dist(**output).log_prob(target)                   # [B, K, D]
        log_probs = log_probs.sum(dim=tuple(range(1, log_probs.dim())))         # [B]                                                            
        log_probs = self.reduce(log_probs)

        return - log_probs
    

    
class ChamferNMLELoss(NMLELoss):
    """ Negative values of log likelihood  \sum_{i=1}^{B}\log p(\theta_i|\mu_i^*, \sigma_i^*)
        Chamfer distance for multi-posterior
     
    """
    def __init__(self, post_dist: dist.Distribution, reduction='mean') -> None:
        super(ChamferNMLELoss, self).__init__(post_dist, reduction)
    

    def forward(self, output, target):
        """ Compute the likelihood 

        Args:
            output [B, K, D, 2]): prediction
            target [B, K, D]: thetas
        """ 
        # Reshape for broadcasting
        output.loc = output.loc.unsqueeze(1)        # [B, 1, K, D]
        output.scale = output.scale.unsqueeze(1)    # [B, 1, K, D]
        target = target.unsqueeze(2)                # [B, K, 1, D]


        log_probs = self.post_dist(**output).log_prob(target).sum(-1)                   # [B, K, K]
        log_probs_1, _ = torch.max(log_probs, 1)                                        # [B, K]
        log_probs_2, _ = torch.max(log_probs, 2)                                        # [B, K]


        # Here we simply use the top 1 values wrt each target point
        log_probs, _ = torch.topk(log_probs, 1, dim=-1)                                 # [B, K]
        log_probs = (log_probs_1.sum(-1) + log_probs_2.sum(-1))  / 2                    # [B]                                                          

        log_probs = self.reduce(log_probs)

        return - log_probs
    


class SetNMLELoss(NMLELoss):
    """ Objective permutation invaraince version of negative values of log likelihood  \sum_{i=1}^{B}\log p(\theta_i|\mu_i^*, \sigma_i^*) """
    def __init__(self, post_dist: dist.Distribution, reduction='mean') -> None:
        super(SetNMLELoss, self).__init__(post_dist, reduction)
    

    def forward(self, output, target):
        """ Compute the likelihood 

        Args:
            output [B, K, D, 2]): prediction
            target [B, K, D]: thetas
        """
        # Permutation Invariance
        # Reshape for broadcasting
        output.loc = output.loc.unsqueeze(1)        # [B, 1, K, D]
        output.scale = output.scale.unsqueeze(1)    # [B, 1, K, D]
        target = target.unsqueeze(2)                # [B, K, 1, D]

        log_probs = self.post_dist(**output).log_prob(target).sum(-1)                   # [B, K, K]

        # lower bound
        log_probs = torch.logsumexp(log_probs, dim=-1)                                  # [B, K]
        log_probs = log_probs.sum(-1)                                                   # [B]                                                          

        return - log_probs.mean()
