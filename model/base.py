import torch
import torch.nn as nn
from attrdictionary import AttrDict

from .embedder import Embedder
from .encoder import Encoder
from .head import OutputHead



class Aline(nn.Module):
    """
    Base ALINE model that consists of an embedder, encoder, and head to process
    input tasks.

    Attributes:
        embedder (Embedder): An embedder module used to convert input tasks into
            embeddings.
        encoder (Encoder): An encoder module that performs the attention mechanism.
        head (OutputHead): A head module that outputs predictions or computes
            log-likelihoods.
    """

    def __init__(
        self, embedder: Embedder, encoder: Encoder, head: OutputHead
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        batch: AttrDict,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model, processing the input batch
        of tasks.

        Args:
            batch (AttrDict): A batch of input tasks to be processed.

        Returns:
            torch.Tensor: The output tensor from the head module, which could be
                predictions or log-likelihoods.
        """
        embedding = self.embedder(batch)
        encoding = self.encoder(batch, embedding)

        return self.head(batch, encoding)
