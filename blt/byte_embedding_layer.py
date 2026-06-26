import torch.nn as nn
from .byte_encoder import UTF8Encoder

class ByteEmbedding(nn.Module):

    def __init__(self, dim):

        super().__init__()

        # 256 possible byte values
        self.embedding = nn.Embedding(
            num_embeddings=256,
            embedding_dim=dim
        )

    def forward(self, x):

        # [seq]
        return self.embedding(x)



# [seq_len, 256]