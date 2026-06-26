import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from blt.byte_encoder import UTF8Encoder
from blt.config import GlobalConfig
from blt.blt_model import ByteLatentTransformer

# Sample text dataset (if dataset.txt is not present)
DEFAULT_CORPUS = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
"""


def main():
    # 1. Device Setup (prefer Apple Silicon MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using training device: {device}")

    # 2. Config setup
    # We use a smaller configuration (e.g. 6 layers, 384 embd) for fast learning in this demo.
    # To run the full 100M parameter model, change to: n_layer=12, n_head=12, n_embd=768.
    config = GlobalConfig(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256
    )

    print("Initializing model...")
    model = ByteLatentTransformer(config).to(device)

    # Calculate parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # 3. Load dataset
    dataset_path = "dataset.txt"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("dataset.txt not found, creating a default Shakespeare sample dataset...")
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CORPUS.strip())
        text = DEFAULT_CORPUS.strip()

    # Encode raw text into bytes
    print("Encoding dataset to bytes...")
    byte_data = UTF8Encoder.encode(text).to(device)
    print(f"Total dataset size: {len(byte_data)} bytes")

    # 4. Training hyperparameters
    epochs = 600  # Increased epochs to memorize the tiny dataset
    lr = 3e-4
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print("Starting training loop...")
    model.train()
    
    # We will segment the training text into overlapping blocks
    block_len = config.block_size
    batch_size = 4  # Process a few random blocks per step
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Random overlapping crops for better learning
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            # Random start index
            idx = torch.randint(0, len(byte_data) - block_len - 1, (1,)).item()
            chunk = byte_data[idx : idx + block_len + 1]
            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])
            
        # For simplicity in this demo model, we process the batch sequentially
        # since our model currently expects unbatched 1D inputs in the patch encoder.
        epoch_loss = 0.0
        for b_in, b_tgt in zip(batch_inputs, batch_targets):
            logits = model(b_in) # Shape [1, block_len, 256]
            loss = F.cross_entropy(logits.view(-1, 256), b_tgt.view(-1))
            loss = loss / batch_size # Normalize loss
            loss.backward()
            epoch_loss += loss.item() * batch_size
            
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {epoch_loss/batch_size:.4f}")

    # 5. Save Model Checkpoint
    checkpoint_path = "blt_checkpoint.pt"
    print(f"Saving checkpoint to {checkpoint_path}...")
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)
    print("Training complete and checkpoint saved!")


if __name__ == "__main__":
    main()
