import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):

        attn_out, _ = self.attn(
            x, x, x,
            attn_mask=attn_mask
        )

        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)

        x = self.norm2(x + ff_out)

        return x

class LocalTransformer(nn.Module):

    def __init__(
        self,
        dim=256,
        layers=4,
        heads=4
    ):

        super().__init__()

        self.blocks = nn.ModuleList([

            TransformerBlock(dim, heads)

            for _ in range(layers)
        ])

    def forward(self, x):

        # x = [batch, seq, dim]
        import torch
        seq_len = x.shape[1]
        
        # Create a causal upper triangular mask to block future positions
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )

        for block in self.blocks:

            x = block(x, attn_mask=attn_mask)

        return x