# Byte Latent Transformer (BLT)

This is an unofficial, from-scratch PyTorch implementation of the **Byte Latent Transformer (BLT)**, based on the paper: [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871).

## Overview

Traditional Large Language Models (LLMs) rely on a tokenizer (like BPE or SentencePiece) to convert text into subword tokens before feeding them to the transformer. The Byte Latent Transformer bypasses tokenization entirely, operating directly on **raw bytes**. 

To avoid the high computational cost of running a standard transformer over very long byte sequences, BLT uses a hierarchical patch-based architecture:
1. **Local Encoder**: Encodes raw bytes and uses an entropy predictor to dynamically group bytes into "patches" (similar to words or subwords, but learned organically).
2. **Global Latent Transformer**: A large transformer that processes these variable-length patches. Because it operates on patches rather than individual bytes, it is extremely efficient and scales well.
3. **Local Decoder**: Takes the latent patch representations and decodes them back into raw bytes autoregressively using cross-attention.

## Project Structure

```text
.
├── blt/                        # Core model package
│   ├── blt_model.py            # Main ByteLatentTransformer class
│   ├── byte_encoder.py         # UTF-8 Byte Encoder
│   ├── patch_builder.py        # Dynamic patching logic
│   ├── entropy_pred.py         # Predicts patch boundaries via entropy
│   ├── local_transformer_block.py # Causal Local Encoder/Decoder blocks
│   ├── global_transformer.py   # Global Latent Transformer
│   ├── config.py               # Model configurations (dimensions, layers, etc.)
│   └── ...                     # Sub-components (attention, MLP, embeddings)
├── train.py                    # Script to train the model from scratch
├── generate.py                 # Script to generate text byte-by-byte
├── dataset.txt                 # Tiny sample dataset (Shakespeare)
└── README.md
```

## Setup & Requirements

- Python 3.8+
- PyTorch (with MPS support for Apple Silicon, or CUDA for NVIDIA)

```bash
pip install torch
```

## Usage

### 1. Training

To train the model from scratch on the sample `dataset.txt`:

```bash
python3 train.py
```

This will run for 300 epochs on a small Shakespeare text to demonstrate that the model can learn and memorize character boundaries. It saves a checkpoint to `blt_checkpoint.pt`.

*Note for Mac Users: If you encounter an OpenMP error during training, run with `KMP_DUPLICATE_LIB_OK=TRUE python3 train.py`.*

### 2. Generation

Once the model is trained, you can generate text byte-by-byte autoregressively. The script uses Top-K sampling for high-quality generation.

```bash
python3 generate.py
```

## Scaling Up

By default, the architecture is configured for a tiny demonstration. You can scale it up to GPT-2 Small/Medium sizes (~120M parameters) by modifying `blt/config.py`:

```python
class GlobalConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
```

*Note: Training a 100M+ parameter byte-level model to fluency requires gigabytes of text data and significant compute (multiple GPUs).*

## License

MIT License
