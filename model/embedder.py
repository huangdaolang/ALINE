import torch
import torch.nn as nn
from typing import Any
from attrdictionary import AttrDict

class Embedder(nn.Module):
    """
    Embedder module that processes batches in three different modes:
    - 'theta': For predicting latent variables
    - 'data': For predicting x tasks
    - 'mix': Combination of tasks and theta prediction
    
    The embedder takes a batch with context, query, and target tasks,
    and produces embeddings in the order: context, query, target.
    """
    
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        n_target_theta: int = 0,
        embedding_type: str = "data",
        **kwargs: Any
    ) -> None:
        """
        Initialize the embedder module
        
        Args:
            dim_x: Dimension of x tasks
            dim_y: Dimension of y tasks
            dim_embedding: Dimension of embedding vectors
            n_theta: Number of latent variables (theta)
            embedding_type: Type of embedding ('data', 'theta', or 'mix')
            T_token: Whether to include a time token in the embeddings
        """
        super().__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_embedding = dim_embedding
        self.n_target_theta = n_target_theta
        self.embedding_type = embedding_type
        
        # Create embedders for x and y
        self.x_embedder = nn.Sequential(
            nn.Linear(dim_x, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )
        
        self.y_embedder = nn.Sequential(
            nn.Linear(dim_y, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )
        
        # Create theta tokens for theta and mix modes
        if embedding_type in ["theta", "mix"]:
            if self.n_target_theta <= 0:
                raise ValueError("dim_theta must be positive for theta or mix embedding type")
            
            # Create learnable theta tokens (one for each dimension of theta)
            self.theta_tokens = nn.Parameter(torch.randn(self.n_target_theta, dim_embedding))
    
    def forward(self, batch: AttrDict) -> torch.Tensor:
        """
        Process batch and generate embeddings based on the selected mode
        
        Args:
            batch: Batch containing context_x, context_y, query_x, and other tasks
                  depending on the embedding mode
        
        Returns:
            embeddings: Tensor of shape [B, N, dim_embedding] with context, query, 
                      and target embeddings concatenated
        """
        batch_size = batch.context_x.shape[0]
        
        # Extract dimensions
        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]

        # Process according to embedding type
        if self.embedding_type == "data":
            embeddings = self._embed_data_mode(batch, batch_size, n_context, n_query)
        elif self.embedding_type == "theta":
            embeddings = self._embed_theta_mode(batch, batch_size, n_context, n_query)
        elif self.embedding_type == "mix":
            embeddings = self._embed_mix_mode(batch, batch_size, n_context, n_query)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

        return embeddings

    def _create_time_embedding(self, batch: AttrDict, batch_size: int) -> torch.Tensor:
        """
        Create a time step token from batch.t

        Args:
            batch: Batch containing t (a single scalar value t/T for the entire batch)
            batch_size: Batch size

        Returns:
            time_embedding: Time step token of shape [B, 1, dim_embedding]
        """
        # t is a scalar for the entire batch (t/T), we need to expand it for the batch
        t_value = batch.t
        if not torch.is_tensor(t_value):
            t_value = torch.tensor([t_value], device=batch.context_x.device, dtype=torch.float)
        else:
            # Ensure t is a scalar or 1D tensor with a single value
            t_value = t_value.view(1)

        # Expand to batch size
        t_batch = t_value.expand(batch_size, 1)  # [B, 1]

        # Embed the time step
        time_embedding = self.time_embedder(t_batch).unsqueeze(1)  # [B, 1, dim_embedding]
        return time_embedding

    def _embed_data_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """
        Data mode embedding: predict x tasks
        - Concatenate context_x, query_x, target_x and embed
        - Only context_y is embedded
        - Context embedding = x_embedding + y_embedding
        - Final embedding = [context_embedding, query_embedding, target_embedding]
        """
        # Embed x (concatenate all x tasks)
        x_all = torch.cat(
            [batch.context_x, 
             batch.query_x,
             batch.target_x], 
            dim=1
        )
        embeddings = self.x_embedder(x_all)  # [B, n_context+n_query+n_target, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        embeddings[:, :n_context] = embeddings[:, :n_context] + y_embeddings_context

        
        return embeddings
    
    def _embed_theta_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """
        Theta mode embedding: predict latent variables
        - Concatenate context_x, query_x and embed
        - Only context_y is embedded
        - Context embedding = x_embedding + y_embedding
        - Create n theta tokens for n dimensions of target_theta
        - Final embedding = [context_embedding, query_embedding, theta_tokens]
        """
        # Embed x (context and query only)
        x_context_query = torch.cat(
            [batch.context_x, 
             batch.query_x], 
            dim=1
        )
        x_embeddings = self.x_embedder(x_context_query)  # [B, n_context+n_query, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        # Add x and y embeddings for context
        context_embeddings = x_embeddings[:, :n_context] + y_embeddings_context
        
        # Extract query embeddings
        query_embeddings = x_embeddings[:, n_context:]
        
        # Create theta tokens (expanded to batch size)
        theta_embeddings = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, dim_theta, dim_embedding]
        
        # Concatenate all embeddings in order: context, query, theta tokens
        embeddings_list = [context_embeddings, query_embeddings, theta_embeddings]

        embeddings = torch.cat(embeddings_list, dim=1)  # [B, N, dim_embedding]
        
        return embeddings
    
    def _embed_mix_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """
        Mix mode embedding: combination of tasks and theta prediction
        - Concatenate context_x, query_x, target_x and embed
        - Only context_y is embedded
        - Context embedding = x_embedding + y_embedding
        - Create theta tokens for target_theta
        - Target embedding = [target_x_embedding, theta_tokens] (concatenated, not added)
        - Final embedding = [context_embedding, query_embedding, target_embedding]
        """
        
        # Embed x (concatenate all x tasks)
        x_all = torch.cat(
            [batch.context_x, 
             batch.query_x,
             batch.target_x], 
            dim=1
        )
        x_embeddings = self.x_embedder(x_all)  # [B, n_context+n_query+n_target, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        # Add x and y embeddings for context
        context_embeddings = x_embeddings[:, :n_context] + y_embeddings_context
        
        # Extract query and target x embeddings
        query_embeddings = x_embeddings[:, n_context:n_context+n_query]
        target_embeddings = x_embeddings[:, n_context+n_query:]
        
        # Simply expand theta tokens to batch size
        theta_embeddings = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, dim_theta, dim_embedding]
        
        # Combine target embeddings by concatenation (not addition)
        # Concatenate target_x_embeddings and theta_embeddings
        embeddings_list = [context_embeddings, query_embeddings, target_embeddings, theta_embeddings]
        
        embeddings = torch.cat(embeddings_list, dim=1)  # [B, N, dim_embedding]
        
        return embeddings