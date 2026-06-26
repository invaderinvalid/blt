import torch
import torch.nn as nn


# config won't be used in this file, but it is passed to the class for future use


class PatchPositionalEmbedding(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.pos_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

    def forward(self, x):

        """
        x = [B,T,C]

        B = batch
        T = patches
        C = embedding dim
        """

        B, T, C = x.shape

        # create positions
        positions = torch.arange(
            T,
            device=x.device
        )

        # get vectors
        pos = self.pos_embedding(
            positions
        )

        # add positional info
        return x + pos
    


