import torch
import torch.nn as nn
import torch.distributions as dist
from attrdictionary import AttrDict

class RandomDesign(nn.Module):
    """ Random Design Policy

    Args:
        dim_x (int): dimension of design
        design_batch (int): number of designs in a trajectory
        random_type (str): distribution type of random sampler
        random_kwargs (dict, optional): dict of args for random sampler
    """
    def __init__(
        self,
        dim_x: int,
        random_type: str,
        random_kwargs: dict = None
    ):
        super(RandomDesign, self).__init__()

        self.design_dim = dim_x
        self.random_type = random_type
        if random_type == "uniform":
            if random_kwargs == None:
                random_kwargs = {'low': torch.tensor([0.0]), 'high': torch.tensor([1.0])}
            self.sampler = dist.Uniform(**random_kwargs)
        elif random_type == "normal":
            if random_kwargs == None:
                random_kwargs = {'loc': torch.tensor([0.0]), 'scale': torch.tensor([1.0])}
            self.sampler = dist.Normal(**random_kwargs)
        else:
            raise ValueError(f"Random design type {random_type} is not supported!")

    def design_candidates(self, batch_size: int = 1, T: int = 30):
        """ Generate a batch of designs

        Args:
            batch_size (int): batch size of parallel experiments B
            T (int): number of design candidates in a trajectory

        Returns:
            designs [B, T, D]
        """
        shape = (batch_size, T, self.design_dim)
        return self.sampler.sample(shape).squeeze(-1)
    
    def forward(self, batch: AttrDict):
        B, t, _ = batch.context_x.shape
        xi = self.sampler.sample((B, self.design_dim)).squeeze(-1)
        idx = torch.zeros((B, 1), dtype=torch.int64).fill_(torch.tensor(t))
        log_prob = self.sampler.log_prob(xi)
        return xi, idx, log_prob


class GridDesign(nn.Module):
    """ Grid Design Policy

    Args:
        dim_x (int): dimension of design
        design_scales (list, optional): list of design scales wrt each dimension
    """
    def __init__(
        self,
        dim_x: int,
        design_scales: list = None,
    ):
        super(GridDesign, self).__init__()

        self.design_dim = dim_x
        if design_scales is not None:
            assert self.design_dim == len(design_scales), f"The length of design scales {len(design_scales)} conflicts with the dimension of design space {self.design_dim}!"
        self.design_scales = design_scales
        

    def design_candidates(self, batch_size=1, num_points=30):
        """ Generate a batch of designs

        Args:
            batch_size (int): batch size of parallel experiments B
            num_points (int): number of points in each dimension

        Returns:
            designs [B, T, D]
        """
        if self.design_scales is None:
            linspaces = [torch.linspace(0, 1, num_points) for _ in range(self.design_dim)]
        else:
            linspaces = [torch.linspace(0, 1, num_points) * design_scale for design_scale in self.design_scales]
        
        self.xi_designs = (
            torch.stack(torch.meshgrid(*linspaces, indexing="ij"), dim=-1)
            .reshape(-1, self.design_dim)
            .unsqueeze(0)
        )   # [1, num_points ** D, D]

        return self.xi_designs.expand(batch_size, -1, -1)
    
    def forward(self, batch: AttrDict):
        B, t, _ = batch.context_x.shape
        xi = self.xi_designs[:, t, :].expand(B, -1)
        idx = torch.zeros((B, 1), dtype=torch.int64).fill_(torch.tensor(t))
        log_prob = torch.ones((B))
        return xi, idx, log_prob
