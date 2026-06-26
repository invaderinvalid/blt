import torch
import torch.nn as nn
from blt.pipeline import BytePatchEncoder
from blt.global_transformer import GlobalTransformer
from blt.config import GlobalConfig


class LocalDecoder(nn.Module):

    def __init__(self, config):
        """
        Causal Local Decoder for next-byte prediction.
        Uses cross-attention over completed patches, followed by causal self-attention over bytes.
        """
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # Learnable dummy patch embedding to avoid NaNs at the beginning of the sequence
        self.dummy_patch = nn.Parameter(torch.zeros(1, 1, config.n_embd))

        # Cross-attention: bytes querying completed patch embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.n_embd)

        # Causal Self-attention over bytes
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.n_embd)

        # Feedforward MLP
        self.ff = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd)
        )
        self.norm3 = nn.LayerNorm(config.n_embd)

        # Final prediction head mapping to byte vocabulary size (256)
        self.pred_head = nn.Linear(config.n_embd, 256)

    def forward(self, byte_embeddings, patch_embeddings, boundaries):
        """
        Args:
            byte_embeddings (Tensor): [1, N, C]
            patch_embeddings (Tensor): [1, M, C]
            boundaries (list): end index of each patch (0-indexed)
        """
        B, N, C = byte_embeddings.shape
        _, M, _ = patch_embeddings.shape

        # Prepend the learnable dummy patch to extended patches
        dummy = self.dummy_patch.expand(B, 1, -1)
        extended_patches = torch.cat([dummy, patch_embeddings], dim=1)  # [B, M+1, C]

        # Construct causal cross-attention mask of shape [N, M+1]
        # extended_boundaries has size M+1, first element is -1 (always completed)
        extended_boundaries = torch.tensor([-1] + boundaries, device=byte_embeddings.device)

        # Vectorized broadcast comparison: extended_boundaries[k] < i
        i_indices = torch.arange(N, device=byte_embeddings.device).unsqueeze(1)  # [N, 1]
        cond = extended_boundaries.unsqueeze(0) < i_indices  # [N, M+1]

        mask = torch.zeros(N, M + 1, device=byte_embeddings.device)
        mask = mask.masked_fill(~cond, float('-inf'))

        # 1. Cross-attention
        cross_out, _ = self.cross_attn(
            query=byte_embeddings,
            key=extended_patches,
            value=extended_patches,
            attn_mask=mask
        )
        x = self.norm1(byte_embeddings + cross_out)

        # 2. Causal Self-attention over bytes
        causal_mask = torch.triu(
            torch.full((N, N), float('-inf'), device=x.device),
            diagonal=1
        )
        self_attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask
        )
        x = self.norm2(x + self_attn_out)

        # 3. MLP
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        # 4. Final projection to byte logits
        logits = self.pred_head(x)  # [1, N, 256]
        return logits


class ByteLatentTransformer(nn.Module):

    def __init__(self, config: GlobalConfig):
        """
        Unified Byte Latent Transformer (BLT) Language Model
        """
        super().__init__()
        self.config = config

        # Local Encoder (Patch Encoder): Processes raw bytes into patch representations
        self.patch_encoder = BytePatchEncoder(dim=config.n_embd)

        # Global Transformer: Performs autoregressive reasoning over patches
        self.global_transformer = GlobalTransformer(config)

        # Local Decoder: Maps patch representations back to byte logit predictions
        self.local_decoder = LocalDecoder(config)

    def forward(self, byte_ids):
        """
        Forward pass for the Byte Latent Transformer.

        Args:
            byte_ids (Tensor): Tensor of raw byte IDs, shape [seq_len].

        Returns:
            Tensor: Logits predicting next-byte distribution at each position,
                    shape [1, seq_len, 256].
        """
        # 1. Encode raw byte sequence into patch embeddings and boundaries
        # patches shape: [num_patches, n_embd]
        patches, boundaries = self.patch_encoder(byte_ids)

        # Extract byte embeddings to query the decoder
        byte_embeddings = self.patch_encoder.embedding(byte_ids).unsqueeze(0)  # [1, seq_len, n_embd]

        # Add batch dimension to patch embeddings: [1, num_patches, n_embd]
        patches_batched = patches.unsqueeze(0)

        # 2. Run Global Transformer over patch sequence
        # global_out shape: [1, num_patches, n_embd]
        global_out = self.global_transformer(patches_batched)

        # 3. Run Local Decoder to get byte-level logits
        # logits shape: [1, seq_len, 256]
        logits = self.local_decoder(byte_embeddings, global_out, boundaries)

        return logits
