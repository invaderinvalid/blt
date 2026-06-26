from torch import nn
import torch
import math
class CausalSelfAttention(nn.Module):    
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

            # [B,T,256] -> [B,T,768]

            # ---------------------------------

            qkv = self.qkv(x)

            # split into q k v

            q, k, v = qkv.split(

                self.n_embd,

                dim=2

            )

            # shapes

            # q = [B,T,256]

            # k = [B,T,256]

            # v = [B,T,256]

            # ---------------------------------

            # STEP 2 : split into heads

            # [B,T,256] -> [B,8,T,32]

            # ---------------------------------

            q = q.view(

                B,

                T,

                self.n_head,

                self.head_dim

            ).transpose(1,2)

            k = k.view(

                B,

                T,

                self.n_head,

                self.head_dim

            ).transpose(1,2)

            v = v.view(

                B,

                T,

                self.n_head,

                self.head_dim

            ).transpose(1,2)

            # shapes now

            # q = [B,8,T,32]

            # k = [B,8,T,32]

            # v = [B,8,T,32]

            # ---------------------------------

            # STEP 3 : attention scores

            # Q × Kᵀ

            # ---------------------------------

            scores = (

                q @ k.transpose(-2,-1)

            )

            # shape

            # [B,8,T,T]

            # ---------------------------------

            # STEP 4 : scale

            # divide by sqrt(head_dim)

            # ---------------------------------

            scores = scores / math.sqrt(

                self.head_dim

            )

            # ---------------------------------

            # STEP 5 : causal mask

            # block future positions

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

            # convert to probabilities

            # ---------------------------------

            attn = torch.softmax(

                scores,

                dim=-1

            )

            # shape

            # [B,8,T,T]

            # return for testing now


            y = attn @ v
            return y

