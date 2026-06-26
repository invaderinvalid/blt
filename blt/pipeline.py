from blt.byte_embedding_layer import ByteEmbedding
from blt.local_transformer_block import LocalTransformer
from blt.entropy_pred import EntropyPredictor
from blt.patch_builder import PatchBuilder
import torch.nn as nn


class BytePatchEncoder(nn.Module):

    def __init__(self, dim=256):

        super().__init__()

        self.embedding = ByteEmbedding(
            dim
        )

        self.local_transformer = LocalTransformer(
            dim=dim
        )

        self.entropy = EntropyPredictor(
            dim
        )

        self.patch_builder = PatchBuilder()

    def forward(self, byte_ids):

        """
        byte_ids = [seq]
        
        Returns:
            patches (Tensor): [num_patches, dim]
            boundaries (list): end indices of patches
        """

        x = self.embedding(
            byte_ids
        )

        # add batch
        x = x.unsqueeze(0)

        local_out = self.local_transformer(
            x
        )

        entropy = self.entropy(
            local_out
        )

        patches, boundaries = self.patch_builder.build(
            local_out.squeeze(0),
            entropy.squeeze(0)
        )

        return patches, boundaries