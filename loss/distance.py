import torch
import torch.nn as nn
import torch.distributions as dist    

class L2Distance(nn.Module):
    """ Expectation of EMD l2 distance between source points and posterior predictions """
    def __init__(self, post_dist: dist.Distribution, N=1000, reduction='mean') -> None:
        super(L2Distance, self).__init__()

        self.post_dist = post_dist
        self.N = N
        self.reduction = reduction
    

    def forward(self, output, target):
        """ Compute the L2 distance 

        Args:
            output (AttriDict): prediction
            target [B, (K, D)]: thetas
        """
        m = self.post_dist(**output)
        if self.post_dist is dist.Normal:
            samples = m.sample((self.N, ))                                      # [N, B, K, D]
            samples = samples.permute(1, 0, 2, 3)                               # [B, N, K, D]
        else: 
            samples = m.sample((self.N, target.shape[-2]))                      # [N, K, B, D]
            samples = samples.permute(2, 0, 1, 3)                               # [B, N, K, D]

        samples, _ = torch.sort(samples, dim=-2)
        target, _ = torch.sort(target.unsqueeze(1), dim=-2)                     # [B, 1, K, D]

        l2 = torch.norm(target - samples, p=2, dim=-1)                          # [B, N, K]
        l2 = torch.sum(l2, -1)                                                  # [B, N]
        l2 = torch.mean(l2, -1)                                                 # [B]
        
        # reduction                                                 
        if self.reduction == "mean":
            l2 = l2.mean()
        elif self.reduction == "sum":
            l2 = l2.sum()

        return l2