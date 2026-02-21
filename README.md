# Aether Byte

## Overview

Aether Byte is an experimental multimodal AI model designed to unify text, images, and tool interactions into a single transformer-based architecture. The core philosophy is to use **bytes as a universal substrate**, treating all inputs and outputs as sequences of bytes. This allows the model to handle diverse data types (e.g., raw text bytes, compressed image latents, tool calls, and special tokens) within a shared embedding space, enabling seamless multimodal reasoning, generation, and agentic behavior.

By tokenizing at the byte level (with a vocabulary of 512, where 0-255 represent literal bytes and higher IDs are special tokens like BOS, EOS, IMG_START, etc.), the model avoids traditional subword tokenization pitfalls and supports arbitrary binary data. This "bytes-first" approach aims to create a flexible, end-to-end system for:
- Multimodal training and inference (text + vision).
- Agentic loops with memory, thought processes, and tool execution.
- Efficient handling of images via VQ-VAE compression.
- Potential extensibility to other modalities (e.g., audio, binaries) as bytes.

Version 3.3 introduces critical fixes for vision encoder training (ensuring gradients flow to the vision components) and preserves features from v3.2 like VQ-VAE, agent tools, and memory retrieval.

The project is built for research and experimentation, emphasizing small-scale, efficient models that can run on consumer hardware.

## Requirements

Aether Byte requires Python 3.8+ and the following dependencies. Install them via pip:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # For CUDA; use CPU version if no GPU
pip install requests pillow numpy
pip install peft  # Optional: for LoRA training
```

- **torch**: Core deep learning framework (tested with 2.0+; supports CUDA for GPU acceleration).
- **requests**: For downloading datasets and web interactions.
- **Pillow (PIL)**: Image processing.
- **numpy**: Numerical operations.
- **peft** (optional): For Low-Rank Adaptation (LoRA) during training.

Additional notes:
- No external APIs or internet access required beyond dataset downloads (COCO val2017 and pico-banana JSONL).
- The model uses ~500MB VRAM on GPU for inference; more for training.
- Tested on Ubuntu/Linux with CUDA 12.1; should work on Windows/Mac with adjustments.

## Architecture

Aether Byte is a transformer-based model with integrated vision and agent capabilities. Key components:

### Core Model (`ByteTransformer`)
- **Embedding Layer**: `nn.Embedding` with vocab_size=512 (bytes 0-255 + special tokens like BOS=256, IMG_START=262).
- **Vision Encoder (`TinyPatchViT`)**: A lightweight ViT-like model that processes 64x64 RGB images into 48 vision tokens (projected to embed_dim=512). Uses patch embedding, transformer encoder layers, and projection.
- **VQ-VAE (`TinyVQVAE`)**: Compresses images into 128 latent tokens (code_dim=8, codebook_size=256) for efficient byte-level representation. Encoder/decoder use simple conv/transpose layers.
- **Transformer Layers**: 8 layers with:
  - Self-Attention (with RoPE, causal masking, and padding support).
  - Cross-Attention for fusing vision context.
  - Feed-Forward (SwiGLU with dropout).
  - Normalization (RMSNorm).
- **Output Head**: Linear projection to vocab_size for next-token prediction.
- **Parameters**: embed_dim=512, num_heads=16, max_seq_len=8192.

### Utilities
- **Segment Packing/Unpacking**: Wraps binary data (e.g., image latents, tool calls) with start/end tokens and length prefixes.
- **Sampling**: Nucleus sampling (top-p) for generation.
- **Memory (`VectorMemory`)**: Stores and retrieves context snippets via cosine similarity on embeddings (capacity=8192).

### Agent System (`AetherAgent`)
- **Context Management**: Segment-aware buffer to maintain history without truncating mid-segment.
- **Inner Loop**: Up to 8 steps for thought/action cycles, including tool calls.
- **Tools**: Limited set (e.g., `web_search`, `browse_page`, `calc`, `time`, `echo`). Tool calls are JSON-encoded bytes wrapped in TOOL_START/END.
- **Vision Integration**: Encodes images from URLs, fuses via cross-attention.

### Training
- **Dataset (`AetherByteDataset`)**: Mix of multimodal (COCO captions + images) and text-only (pico-banana SFT data). Supports batched vision context.
- **Loss**: Cross-entropy on shifted tokens, with VQ-VAE pre-training.
- **Optimizer**: AdamW with gradient clipping and AMP.

The architecture emphasizes efficiency: small image sizes, tiny VQ-VAE, and byte-level tokens reduce compute needs while enabling unified multimodal handling.

## Installation Guide

1. **Clone the Repository**:
   ```
   git clone https://github.com/your-repo/aether-byte.git  # Replace with actual repo URL
   cd aether-byte
   ```

2. **Install Dependencies**:
   Run the pip commands from the [Requirements](#requirements) section.

3. **Download Datasets (Automatic)**:
   - During training, the script downloads COCO val2017 (~1GB images + annotations) and pico-banana JSONL if not present.
   - Ensure internet access for first run; datasets are cached locally.

4. **Verify Setup**:
   - Run `python aetherbytev3.3.py --mode=agent` to start the interactive agent.
   - If using GPU, confirm with `torch.cuda.is_available()` in a Python shell.

5. **Optional: Pre-trained Checkpoint**:
   - The script loads `./checkpoints/aether_byte_v3.3.pth` if available. Train to generate one, or download from releases (if provided).

## Deployment

Aether Byte can run in two modes: training or agent inference.

### Training Mode
- Command: `python aetherbytev3.3.py --mode=train --lora` (optional LoRA for efficient fine-tuning).
- Outputs: Saves checkpoint to `./checkpoints/aether_byte_v3.3.pth`.
- Duration: ~Hours on GPU (RTX 3060) for 20 epochs on 10k samples.
- Hardware: GPU recommended (at least 4GB VRAM); CPU fallback is slow.

### Agent Mode (Inference)
- Command: `python aetherbytev3.3.py --mode=agent`.
- Interactive CLI: Input text or image URLs; model responds with text, generated images (saved to `./generated_images/`), or tool results.
- Example Usage:
  - Text: "Hello!"
  - Image: "https://example.com/image.jpg" (encodes and describes).
- Hardware: Runs on CPU (~1-2s/token) or GPU (~0.1s/token).
- Deployment Options:
  - **Local**: As-is for development.
  - **Web/Server**: Wrap in Flask/FastAPI for API (e.g., endpoint for user input).
  - **Docker**: Create a Dockerfile for portability:
    ```
    FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    RUN pip install requests pillow numpy peft
    COPY . /app
    WORKDIR /app
    CMD ["python", "aetherbytev3.3.py", "--mode=agent"]
    ```
  - Build and run: `docker build -t aether-byte . && docker run -it aether-byte`.

Note: No production hardening; use for prototyping only.

## Current Limitations

- **Model Scale**: Small size (512 dim, 8 layers) limits performance; hallucinations, incoherence, or poor generalization common.
- **Image Handling**: Fixed 64x64 resolution; low-quality VQ-VAE reconstructions; generation/decoding may fail.
- **Dataset Constraints**: Limited to 10k samples (50% multimodal); no diverse/large-scale training data.
- **Tools**: Only 5 basic tools; no advanced integrations (e.g., no full web browsing, code execution).
- **Safety**: Basic censorship (e.g., regex on harmful commands); no robust alignment.
- **Performance**: Slow on CPU; max_seq_len=8192 but context trimming is naive.
- **Vision Training**: Fixed in v3.3, but still experimental—gradients flow, but efficacy unproven at scale.
- **Extensibility**: Bytes substrate is promising but untested for other modalities (e.g., audio).
- **Dependencies**: Relies on external downloads; potential issues with dataset availability.
- **No Evaluation**: Lacks metrics, benchmarks, or fine-tuning scripts.

Future work could scale up, add modalities, or integrate larger datasets.

Licence: Free for everyone.

Author: Tasmanian Shack Dweller 
