import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
from typing import Optional


class EfficientTransformerEncoderLayer(TransformerEncoderLayer):
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
    ) -> Tensor:
        # Calculate number of context elements and other dimensions
        batch_size, seq_len, embed_dim = x.shape
        num_ctx = torch.sum(attn_mask[0, :] == 0).item()

        # Split into context and non-context parts
        context = x[:, :num_ctx, :]
        non_context = x[:, num_ctx:, :]

        # Process context with self-attention
        context_out = self.self_attn(
            context, context, context,
            attn_mask=None,
            key_padding_mask=key_padding_mask[:, :num_ctx] if key_padding_mask is not None else None,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        # Process non-context with attention to the full sequence based on the mask
        non_context_out = self.self_attn(
            non_context,  # Query: non-context elements
            x,  # Key: full sequence
            x,  # Value: full sequence
            attn_mask=attn_mask[num_ctx:, :],  # Use the relevant part of the attention mask
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        # Combine results
        x_out = torch.cat([context_out, non_context_out], dim=1)

        return self.dropout1(x_out)

class Encoder(nn.Module):
    """
    Encoder module that processes batches in three different modes:
    - 'theta': For predicting latent variables only
    - 'data': For predicting data only
    - 'mix': Combination of data and theta prediction
    
    """
    def __init__(
            self,
            dim_embedding,
            dim_feedforward,
            n_head,
            dropout,
            num_layers,
    ):
        """
        Initialize the encoder module
        
        Args:
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            n_head: Number of attention heads   
            dropout: Dropout rate
            num_layers: Number of layers
        """
        super().__init__()
        # Create the encoder layer
        encoder_layer = EfficientTransformerEncoderLayer(
            dim_embedding, n_head, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def create_mask(self, batch):
        """
        Create a mask for the encoder
        
        Args:
            batch: Batch containing context_x, query_x, and target_all

        Returns:
            mask: Mask for the encoder
        """
        # Get base dimensions
        num_context = batch.context_x.shape[1]
        num_query = batch.query_x.shape[1]
        num_target = batch.target_all.shape[1]
        num_all = num_context + num_query + num_target

        # Create mask where all positions initially can't attend to any position
        mask = torch.zeros(num_all, num_all, device=self.device).fill_(float("-inf"))

        query_start = num_context
        query_end = query_start + num_query
        target_start = query_end

        # All positions can attend to context positions
        mask[:, :num_context] = 0.0

        # Query points can attend to target points based on target_mask
        if hasattr(batch, 'target_mask') and batch.target_mask is not None:
            # Get the target mask (same for all batch elements)
            target_mask = batch.target_mask  # Shape: [num_target]

            # Find indices of selected targets
            selected_targets = torch.where(target_mask)[0]

            # Map selected target indices to positions in the full sequence
            target_positions = selected_targets + target_start

            # Enable attention from all queries to all selected targets at once
            mask[query_start:query_end, target_positions] = 0.0
        else:
            # Default behavior: all queries don't attend to any target
            mask[query_start:query_end, target_start:] = float("-inf")

        return mask

    def forward(self, batch, embeddings):
        """
        Forward pass through the encoder
        
        Args:
            batch: Batch containing context_x, query_x, and target_all
            embeddings: Embeddings for the encoder

        Returns:
            out: Output of the encoder
        """
        mask = self.create_mask(batch)
        out = self.encoder(embeddings, mask=mask)
        return out


class EncoderWithTime(nn.Module):
    def __init__(
        self,
        dim_embedding,
        dim_feedforward,
        n_head,
        dropout,
        num_layers,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            dim_embedding, n_head, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
    def create_mask(self, batch):
        # Get base dimensions
        num_context = batch.context_x.shape[1]
        num_query = batch.query_x.shape[1]
        num_target = batch.target_all.shape[1]

        has_time = hasattr(batch, 't') and batch.t is not None


        num_global = 1 if has_time else 0

        num_all = num_global + num_context + num_query + num_target

        # Create mask where all positions initially can't attend to any position
        mask = torch.zeros(num_all, num_all, device=self.device).fill_(float("-inf"))

        # Calculate indices with time token offset
        if has_time:
            context_start = 1
            context_end = 1 + num_context
        else:
            context_start = 0
            context_end = num_context
        query_start = context_end
        query_end = query_start + num_query
        target_start = query_end

        # Context positions can attend to themselves (self-attention)
        for i in range(context_start, context_end):
            for j in range(context_start, context_end):
                mask[i, j] = 0.0

        # All positions can attend to context positions
        mask[:, context_start:context_end] = 0.0

        # Query positions can attend to the time token (if it exists)
        if has_time:
            # Query positions can attend to time token
            mask[query_start:query_end, 0] = 0.0

        # Query points can attend to target points based on target_mask
        if hasattr(batch, 'target_mask') and batch.target_mask is not None:
            # Get the target mask (same for all batch elements)
            target_mask = batch.target_mask  # Shape: [num_target]

            # Find indices of selected targets
            selected_targets = torch.where(target_mask)[0]

            # Map selected target indices to positions in the full sequence
            target_positions = selected_targets + target_start

            # Enable attention from all queries to all selected targets at once
            mask[query_start:query_end, target_positions] = 0.0
        else:
            # Default behavior: all queries can attend to all targets
            mask[query_start:query_end, target_start:] = 0.0

        return mask

    def forward(self, batch, embeddings):
        mask = self.create_mask(batch)
        out = self.encoder(embeddings, mask=mask)
        return out


# Test the implementation
if __name__ == "__main__":
    import time

    def test_efficient_transformer_with_fixed_weights():
        # Parameters
        batch_size = 2
        ctx_size = 30
        query_size = 200
        target_size = 100
        embed_dim = 8
        n_head = 2

        # Create input data
        x = torch.randn(batch_size, ctx_size + query_size + target_size, embed_dim)

        # Create a target mask where only certain targets can be attended to
        target_mask = torch.zeros(target_size, dtype=torch.bool)
        target_mask[0] = True  # First target can be attended to
        target_mask[2] = True  # Third target can be attended to

        # Create a batch object
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self

        batch = AttrDict({
            'context_x': torch.randn(batch_size, ctx_size, 2),
            'query_x': torch.randn(batch_size, query_size, 2),
            'target_all': torch.randn(batch_size, target_size, 1),
            'target_mask': target_mask
        })

        # Create standard encoder layer
        standard_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=32,
            dropout=0.0,  # Use 0 dropout for deterministic behavior
            batch_first=True
        )


        custom_layer = EfficientTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=32,
            dropout=0.0,  # Use 0 dropout for deterministic behavior
            batch_first=True
        )

        # IMPORTANT: Copy weights from standard layer to custom layer
        # This ensures both layers have identical parameters
        custom_layer.load_state_dict(standard_layer.state_dict())

        # Create both encoders using their respective layers
        standard_encoder = nn.TransformerEncoder(standard_layer, num_layers=1)
        custom_encoder = nn.TransformerEncoder(custom_layer, num_layers=1)

        # Create mask function
        def create_mask(batch):
            # Get base dimensions
            num_context = batch.context_x.shape[1]
            num_query = batch.query_x.shape[1]
            num_target = batch.target_all.shape[1]
            num_all = num_context + num_query + num_target

            # Create mask where all positions initially can't attend to any position
            mask = torch.zeros(num_all, num_all).fill_(float("-inf"))

            # All positions can attend to context positions
            mask[:, :num_context] = 0.0

            # Query points can attend to target points based on target_mask
            if hasattr(batch, 'target_mask') and batch.target_mask is not None:
                # Find indices of selected targets
                selected_targets = torch.where(batch.target_mask)[0]

                # Map selected target indices to positions in the full sequence
                target_positions = selected_targets + num_context + num_query

                # Enable attention from all queries to all selected targets
                mask[num_context:num_context + num_query, target_positions] = 0.0
            else:
                # Default behavior: all queries can attend to all targets
                mask[num_context:num_context + num_query, num_context + num_query:] = 0.0

            return mask

        # Create masks
        attention_mask = create_mask(batch)

        # Process inputs with both encoders
        with torch.no_grad():
            # Standard encoder
            start = time.time()
            standard_out = standard_encoder(x, mask=attention_mask)
            print(f"Standard encoder time: {time.time() - start}")

            # Custom encoder
            start = time.time()
            custom_out = custom_encoder(x, mask=attention_mask)
            print(f"Custom encoder time: {time.time() - start}")

        # Check if outputs are close
        outputs_close = torch.allclose(standard_out, custom_out, rtol=1e-4, atol=1e-4)
        print(f"Outputs are {'close' if outputs_close else 'different'}")

        if not outputs_close:
            diff = (standard_out - custom_out).abs()
            print(f"Max difference: {diff.max().item()}")
            print(f"Mean difference: {diff.mean().item()}")

        return standard_out, custom_out

    standard_out, efficient_out = test_efficient_transformer_with_fixed_weights()