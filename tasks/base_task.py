import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Optional, Dict, Tuple, List, Any, Union
from attrdictionary import AttrDict


### generate the experiment outcomes
class Task(nn.Module):
    """
    Base class for all tasks that provides common functionality:
    - Converting between normalized and real design spaces
    - Sampling batches with context, query, and target points
    - Updating batches when query points are selected
    - Supporting different modes: data, theta, mix
    """
    
    def __init__(
        self,
        dim_x: int = 2,
        dim_y: int = 1,
        dim_theta: int = 0,
        mode: str = "data",
        design_scale: float = 1.0,
        outcome_scale: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_theta = dim_theta
        self.mode = mode
        self.design_scale = design_scale
        self.outcome_scale = outcome_scale
        self.device = device
        
        # Validate mode and dimension consistency
        if mode in ["theta", "mix"] and dim_theta <= 0:
            raise ValueError(f"dim_theta must be positive for mode '{mode}'")
    
    @torch.no_grad()
    def sample_theta(self, size, **kwargs):
        """
        Sample theta values from the prior distribution.
        Must be implemented by child classes for theta and mix modes.
        
        Args:
            size: Size of theta batch to generate
            
        Returns:
            theta: Sampled theta values [size, dim_theta]
        """
        raise NotImplementedError("Child classes must implement sample_theta")
    
    def normalise_design(self, x):
        """Convert from real to normalized design space."""
        return x / self.design_scale
    
    def unnormalise_design(self, x):
        """Convert from normalized to real design space."""
        return x * self.design_scale
        
    def normalise_outcomes(self, y):
        """Normalize outcomes for consistent loss calculation."""
        return y / self.outcome_scale
    
    @torch.no_grad()
    def forward(self, xi, theta):
        """
        Forward function that maps xi (design) and theta to y (observation).
        
        Args:
            xi [B, D]: normalized design
            theta [B, K, D]: latent variables to learn from the experiments
            
        Returns:
            observations: [B, 1]
        """
        raise NotImplementedError("Child classes must implement forward")
    
    def log_likelihood(self, y, xi, theta):
        """
        Log likelihood from gaussian noise
        
        Args:
            y [B, 1]
            xi [B, D]: real designs
            theta [B, K, D]
            
        Returns:
            log_prob: log likelihoods, [B, 1]
        """
        raise NotImplementedError("Child classes must implement log_likelihood")
    
    def update_batch_query(self, query, idx):
        """ Update the batch of query points

        Args:
            query: Current query points [B, N, D]
            idx: Indices of query points to remove
        Returns:
            updated_query: Updated query points with selected points removed
        
        """
        B, Nt, D = query.shape
        mask = torch.ones((B, Nt), dtype=torch.bool)
        mask[torch.arange(B).unsqueeze(1), idx] = False
        query = query[mask].view(B, -1, D)
        return query


    def update_batch_context(self, context, new):
        """ Update the batch of context points

        Args:
            context: Current context points [B, N, D]
            new: New context points [B, 1, D]
        Returns:
            updated_context: Updated context points with new points added
        """
        context = torch.cat([context, new], dim=1)
        return context
    
    def update_batch(self, batch, idx):
        """ Update the batch of data
        
        Args:
            batch: Current batch
            idx: Indices of selected query points
            
        Returns:
            updated_batch: Updated batch with selected query points moved to context
        """

        # get the next data points
        next_x = torch.gather(batch.query_x, 1, idx.unsqueeze(2).expand(-1, 1, self.dim_x))
        next_y = torch.gather(batch.query_y, 1, idx.unsqueeze(2).expand(-1, 1, self.dim_y))

        batch.query_x = self.update_batch_query(batch.query_x, idx)
        batch.query_y = self.update_batch_query(batch.query_y, idx)

        batch.context_x = self.update_batch_context(batch.context_x, next_x)
        batch.context_y = self.update_batch_context(batch.context_y, next_y)

        return batch