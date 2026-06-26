import torch
from blt.byte_encoder import UTF8Encoder
from blt.config import GlobalConfig
from blt.blt_model import ByteLatentTransformer


def get_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    # 1. Initialize scaled config to match ~100M-120M parameters
    config = GlobalConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=512
    )

    print("Initializing scaled Byte Latent Transformer (BLT)...")
    model = ByteLatentTransformer(config)
    model.eval()

    # 2. Print parameter counts
    total, trainable = get_parameter_count(model)
    print(f"Total Parameters: {total:,} ({total / 1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable:,}")

    # 3. Encode sample text into byte IDs
    sample_text = (
        "In the Byte Latent Transformer (BLT) architecture, standard tokenization is bypassed."
    )
    print(f"\nInput Text: '{sample_text}'")

    byte_ids = UTF8Encoder.encode(sample_text)
    num_bytes = len(byte_ids)
    print(f"Encoded Sequence: {num_bytes} bytes")

    # 4. Perform the forward pass with default threshold (3.5)
    print("\n--- Running forward pass ---")
    with torch.no_grad():
        logits = model(byte_ids)
    
    print(f"Output logits shape: {logits.shape}")

    # Assertion check: logits should have shape [1, num_bytes, 256]
    # because we make predictions at every single byte position!
    assert logits.shape == (1, num_bytes, 256), f"Unexpected shape {logits.shape}"
    print("\nVerification successful! The scaled BLT runs successfully end-to-end and makes predictions for all bytes.")


if __name__ == "__main__":
    main()
