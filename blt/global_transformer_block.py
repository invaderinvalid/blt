from torch import nn
from blt.causal_attention import CausalSelfAttention
from blt.mlp import MLP


class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        # first normalization
        self.ln1 = nn.LayerNorm(
            config.n_embd
        )

        # attention
        self.attn = CausalSelfAttention(
            config
        )

        # second normalization
        self.ln2 = nn.LayerNorm(
            config.n_embd
        )

        # feed forward
        self.mlp = MLP(
            config
        )

    def forward(self, x):

        # =========================
        # Attention block
        # =========================

        x = x + self.attn(

            self.ln1(x)

        )

        # =========================
        # Feed forward block
        # =========================

        x = x + self.mlp(

            self.ln2(x)

        )

        return x