import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # size of each head
        self.head_dim = (
            config.n_embd // config.n_head
        )

        # one projection creates Q K V together
        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )

        # output projection
        self.proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

    def forward(self, x):

        """
        x : [B,T,C]

        B = batch
        T = sequence length (patches)
        C = embedding dimension

        """

        B, T, C = x.shape

        # ---------------------------------
        # STEP 1 : create qkv
        # ---------------------------------

        qkv = self.qkv(x)

        # split into q k v
        q, k, v = qkv.split(
            self.n_embd,
            dim=2
        )

        # shapes: q, k, v = [B,T,256]

        # ---------------------------------
        # STEP 2 : split into heads
        # ---------------------------------

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        # shapes now: [B,8,T,32]

        # ---------------------------------
        # STEP 3 : attention scores
        # ---------------------------------

        scores = q @ k.transpose(-2, -1)

        # shape: [B,8,T,T]

        # ---------------------------------
        # STEP 4 : scale
        # ---------------------------------

        scores = scores / math.sqrt(self.head_dim)

        # ---------------------------------
        # STEP 5 : causal mask
        # ---------------------------------

        mask = torch.tril(
            torch.ones(
                T,
                T,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # ---------------------------------
        # STEP 6 : softmax
        # ---------------------------------

        attn = torch.softmax(
            scores,
            dim=-1
        )

        # shape: [B,8,T,T]

        # compute attention output
        y = attn @ v  # [B, 8, T, head_dim]

        # transpose back to [B, T, n_head, head_dim] and reshape to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.proj(y)

        return y