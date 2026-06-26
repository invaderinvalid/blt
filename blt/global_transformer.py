import torch.nn as nn
from .positional_encoding import PatchPositionalEmbedding
from .global_transformer_block import TransformerBlock


class GlobalTransformer(nn.Module):

    def __init__(self, config):

        super().__init__()

        # positional embedding
        self.pos_embedding = PatchPositionalEmbedding(
            config
        )

        # stack transformer blocks
        self.blocks = nn.ModuleList([

            TransformerBlock(config)

            for _ in range(
                config.n_layer
            )

        ])

        # final normalization
        self.ln_f = nn.LayerNorm(
            config.n_embd
        )

    def forward(self, x):

        """
        x = patch embeddings

        shape:
        [B,T,C]
        """

        # ============================
        # STEP 1 : add positions
        # ============================

        x = self.pos_embedding(x)

        # ============================
        # STEP 2 : pass through all blocks
        # ============================

        for block in self.blocks:

            x = block(x)

        # ============================
        # STEP 3 : final layer norm
        # ============================

        x = self.ln_f(x)

        return x