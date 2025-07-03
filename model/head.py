import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.distributions.utils import broadcast_all
from typing import Optional, Dict, Any, List, Tuple
from attrdictionary import AttrDict


class AcquisitionHead(nn.Module):
    def __init__(self, dim_embedding: int, dim_feedforward: int, time_token: bool, **kwargs: Any) -> None:
        super().__init__()
        
        if time_token:
            dim_embedding += 1  # add time token to embedding

        self.predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
            nn.Flatten(start_dim=-2),
            nn.Softmax(dim=-1)
        )

    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Embedding feature tensor [B, N, dim_embedding]
        Returns:
            acquisition_scores: Acquisition scores [B, N]
        """
        return self.predictor(z)


class GMMTargetHead(nn.Module):
    def __init__(
        self,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int,
        single_head: bool = False,
        std_min: float = 1e-4,
        **kwargs: Any
        ) -> None:
        
        super().__init__()
        self.dim_embedding = dim_embedding
        self.dim_feedforward = dim_feedforward
        self.dim_y = dim_y
        self.single_head = single_head
        self.num_components = num_components
        self.heads = self.initialize_head(self.dim_embedding, self.dim_feedforward, self.dim_y, self.single_head,
                                          num_components)

        self.std_min = std_min
        # TODO: support multi-output case

    def forward(self, batch: AttrDict, z_target: torch.Tensor) -> AttrDict:

        # Iterate over each head to get their outputs
        if self.single_head:
            output = self.heads(z_target)
            if self.num_components == 1:
                raw_mean, raw_std = torch.chunk(output, 2, dim=-1)
                raw_weights = torch.ones_like(raw_mean)
            else:
                raw_mean, raw_std, raw_weights = torch.chunk(output, 3, dim=-1)
        else:
            outputs = [head(z_target) for head in self.heads]  # list of [B, n_target, dim_y * 3] * components
            raw_mean, raw_std, raw_weights = self._map_raw_output(outputs)

        mean = raw_mean
        std = F.softplus(raw_std) + self.std_min
        weights = F.softmax(raw_weights, dim=-1)

        outs = AttrDict()

        # also outputs params
        outs.mixture_means = mean
        outs.mixture_stds = std
        outs.mixture_weights = weights

        return outs
    
    def initialize_head(self,
                        dim_embedding: int,
                        dim_feedforward: int,
                        dim_outcome: int,
                        single_head: bool,
                        num_components: int,
                        ) -> nn.Module:
        """
        Initializes the model with either a single head or multiple heads based on the `single_head` flag.

        Parameters:
        - dim_embedding (int): The input dimension.
        - dim_feedforward (int): The dimension of the feedforward network.
        - dim_outcome (int): The output dimension.
        - single_head (bool): Flag to determine whether to initialize a single head or multiple heads.
        - num_components (int): The number of components if `single_head` is False.

        Returns:
        - nn.Module: The initialized model head(s).
        """
        if single_head & num_components > 1:
            output_dim = dim_outcome * 3
        else:
            output_dim = dim_outcome * 2

        if single_head:
            model = nn.Sequential(
                nn.Linear(dim_embedding, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, output_dim),
            )
        else:
            model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim_embedding, dim_feedforward),
                        nn.ReLU(),
                        nn.Linear(dim_feedforward, dim_outcome * 3),
                    )
                    for _ in range(num_components)
                ]
            )
        return model

    @staticmethod
    def compute_ll(value: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Computes log-likelihood loss for a Gaussian mixture model.
        """
        components = Normal(means, stds, validate_args=False)
        log_probs = components.log_prob(value)
        weighted_log_probs = log_probs + torch.log(weights)
        return torch.logsumexp(weighted_log_probs, dim=-1)

    @staticmethod
    def _map_raw_output(outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concatenated = torch.stack(outputs).movedim(0, -1).flatten(-2, -1) # [B, T, dim_y * 3 * components]
        raw_mean, raw_std, raw_weights = torch.chunk(concatenated, 3, dim=-1) # 3 x [B, T, components]
        return raw_mean, raw_std, raw_weights



class OutputHead(nn.Module):
    """
    Combined head that processes batches and routes to acquisition and target heads.
    Similar to DPTNP's forward method, it splits input into query and posterior parts.
    """
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int = 10,
        single_head: bool = False,
        std_min: float = 1e-4,
        time_token: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.time_token = time_token
        
        # Acquisition head for design selection
        self.acquisition_head = AcquisitionHead(
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            time_token=time_token,
        )
        
        # Target head for posterior prediction
        self.target_head = GMMTargetHead(
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            num_components=num_components,
            single_head=single_head,
            std_min=std_min
        )
    
    def forward(self, batch: AttrDict, z: torch.Tensor) -> AttrDict:
        """
        Process batch by splitting into context, query and target parts
        
        Args:
            batch: Batch containing context and target tasks
            z: Embedding tensor [B, N, dim_embedding] where N = n_context + n_query + n_target
                Context embeddings are on the left, query in the middle, target on the right
            
        Returns:
            outs: AttrDict containing acquisition and posterior prediction results
        """
        batch_size = z.shape[0]
        
        # Get dimensions from batch
        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]
        
        # Extract query and target embeddings
        z_query = z[:, n_context:n_context+n_query]
        z_target = z[:, n_context+n_query:]  # embeddings of target data + target theta
        
        # Use acquisition head for query selection (design point)
        if self.time_token:
            time_info = batch.t.unsqueeze(1).unsqueeze(1).expand(batch_size, n_query, 1)  # [B, n_query, 1]
            z_query_with_time = torch.cat([z_query, time_info], dim=-1)
            zt = self.acquisition_head(z_query_with_time)  # [B, n_query]
        else:
            zt = self.acquisition_head(z_query)  # [B, n_query]
        
        # Select design with the highest probability during inference, sample during training
        if self.training:
            # Choose design with probabilities zt
            m_design = Categorical(zt)
            idx_next = m_design.sample()  # [B]
            log_prob = m_design.log_prob(idx_next)
        else:
            # Choose design with the largest probability
            log_prob, idx_next = torch.max(zt, -1)
            log_prob = torch.log(log_prob)
        
        # Get the selected design point
        idx_next = idx_next.unsqueeze(1) # [B, 1]
        
        # Use target head for posterior prediction
        posterior_out = self.target_head(batch, z_target)
        posterior_out_query = self.target_head(batch, z_query)  # For ACE uncertainty sampling baseline
        
        # Combine results
        outs = AttrDict(
            posterior_out_query=posterior_out_query,
            posterior_out=posterior_out,
            design_out=AttrDict(
                idx=idx_next,       # [B, 1]
                log_prob=log_prob,  # [B]
                zt=zt               # [B, N_query]
            ),
        )
        return outs