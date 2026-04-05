# MAT Pseudo-Grid LWM: Method 2 — 8×16 Pseudo-Grid Approach

A hybrid foundation model that preserves **LWM's 128-patch tokenization** while enabling **MAT's 2D mask-aware spatial attention** via an 8×16 pseudo-grid reshape. 

## Architecture Overview

```text
Input (B, 2, 32, 32)         ← Real + Imaginary channels
         │
   LWM Patching              ← Flatten + slice into 128 patches of length 16
   (B, 128, 16)
         │
   ┌─ MCM Masking (15%) ─┐   ← Symmetric: same positions in real & imag halves
   │  (B, 128, 16)       │   ← Patches zeroed out
   │  + mask (B, 128)    │   ← Validity mask generated
   └─────────────────────┘
         │
   Linear(16→64) + PosEmb    ← Patch embedding + learnable positional encoding
   (B, 128, 64)
         │
   ┌─ 2D Bridge ─────────┐
   │ reshape (B,128,64)  │   ← Permute + view into pseudo-grid
   │   → (B,64,8,16)     │
   │ mask → (B,1,8,16)   │
   └─────────────────────┘
         │
   ┌─ 3-Stage ATB Body ─────────────┐
   │ Stage 1: ATB (4×4)             │
   │   │                            │
   │ ┌─┴─ Dynamic Mask Relaxation ┐ │
   │ │ Mask MaxPool2d (Downsample)│ │ ← Downsample mask to find any valid tokens in window
   │ │ Repeat Interleave (Upsample) │ ← Upsample mask back to (B,1,8,16)
   │ └─┬──────────────────────────┘ │
   │   │                            │
   │ Stage 2: ATB (4×4) Shifted     │ ← Uses relaxed upsampled mask
   │ Stage 3: ATB (4×4) Standard    │
   └────────────────────────────────┘
         │
   Reverse 2D Bridge         ← (B,64,8,16) → (B,128,64)
   (B, 128, 64)
         │
   ┌─────┴──────────────────────────────────────┐
   │                                            │
   torch.gather(masked_pos)              Inference Only:
   → (B, n_masks, 64)                  ┌──────┴──────┐
   → Linear(64→16)                  GAP→(B,64)    (B,128,64)
   → (B, n_masks, 16)            CLS embedding  Channel embedding
   │
   MSE Loss vs raw patches
```

## Detailed Layer-by-Layer Architecture & Design Choices

### 1. Input Patching (`channels_to_patches`)
**What it does:** The model takes an input of shape `(B, 2, 32, 32)`, representing real and imaginary wireless channel matrices. It flattens the spatial dimensions to `(B, 2, 1024)`, concatenates the real and imaginary parts to form a 2048-length 1D array, and then reshapes it into 128 non-overlapping patches, each of length 16 `(B, 128, 16)`.
**Why this choice:** This exactly mimics the tokenization strategy of LWM v1.1. Maintaining 128 patches of length 16 ensures 100% downstream compatibility with pre-existing LWM infrastructure. Rather than operating on 2D images directly, this converts the wireless matrix into a sequence.

### 2. Symmetric Layer Masking (`mask_patches`)
**What it does:** It randomly drops 15% of the patches, replacing them with zeroes and producing a binary mask. It applies this symmetrically—meaning if patch `i` (in the real half) is masked, patch `i + 64` (in the imaginary half) is also masked.
**Why this choice:** Wireless channels rely heavily on their complex representation (Real + Imaginary). Masking real and imaginary components independently would break the physics-governed continuous complex structure during reconstruction. Symmetric masking forces the network to infer the complete complex token from surrounding context.

### 3. Patch Embedding & Positional Encoding (`patch_proj` and `pos_embed`)
**What it does:** Projects the 16-dimensional patches to the model's inner dimension `d_model` (e.g., 64) via an `nn.Linear` layer mapping it to `(B, 128, 64)`. A learnable positional embedding `pos_embed` is then added.
**Why this choice:** Standard transformer paradigm. Since self-attention is permutation invariant, positional encoding allows the pseudo-grid to understand relative and absolute patch positions. Learnable embeddings were chosen over trigonometric because wireless components do not always behave as contiguous visual regions.

### 4. 2D Spatial Bridge
**What it does:** Uses PyTorch's `view` and `permute` to reshape the 1D sequence of `(B, 128, 64)` into a 2D pseudo-grid of `(B, 64, 8, 16)`. 
**Why this choice:** This is the core "trick" of this method. While standard LWM processes sequences as 1D structures, mapping 128 tokens cleanly into an 8x16 spatial grid permits the usage of 2D Vision Transformer methodologies (like 2D shifted window attention).

### 5. 3-Stage Adjusted Transformer Block (ATB) Body
**What it does:** The backbone consists of 3 ATB stages (Window Size 4x4) processing the 8x16 grid.
**Layer breakdown within each ATB:**
- **WindowMHA (Mask-Aware Attention):** Multi-Head Attention operating on non-overlapping windows. Invalid (masked) tokens receive a `-tau` (e.g., -100) bias penalty, effectively pushing their softmax probability to zero.
- **Concat + 1×1 Conv Fusion:** Attention output is concatenated with the input via a skip connection, then fused with a `1x1` convolution.
- **MLP:** A two-layer feedforward network with GELU activation and a residual connection. 
- **Local Conv:** A 3x3 depthwise-style local convolution added in parallel to the MLP.
**Why these choices:** 
1. **Mask-Awareness:** Including the `-100` penalty ensures that the model only attends to valid tokens, a critical feature adapted from the MAT paper for robust representation learning.
2. **Shifted Windows:** Stage 2 shifts windows diagonally. Because the grid is small (8x16 grid), a 4x4 window enables exactly 8 complete windows. Shifted windows allow cross-window boundary information flow avoiding grid-isolation without requiring global self-attention (which scales quadratically).
3. **Local Conv:** WindowMHA acts globally within a window, but a simple 3x3 local convolution accurately restores high-frequency local spatial correlations that attention mechanisms often smooth over.
4. **No LayerNorm:** Normalization layers inherently use statistics (mean/variance) over the entire batch/sequence. Following the MAT design, LayerNorm is omitted to prevent "leaking" information about masked tokens across unmasked positions.

### 6. Reverse Bridge & Selective Decoder
**What it does:** Reshapes the 2D grid `(B, 64, 8, 16)` back to the 1D sequence `(B, 128, 64)`. Using `torch.gather`, it selectively isolates the embeddings at the original masked indices. These masked embeddings are decoded using a linear projection to reconstruct the 16-length raw patches.
**Why this choice:** Gathering only masked tokens drastically reduces computation. Like a Masked Autoencoder (MAE), the Mean Squared Error (MSE) loss is only computed over missing data, encouraging the model to perform strong extrapolation rather than just identity matching the visible pixels.

### 7. Inference and Embedding Generarion (`gen_raw=True`)
**What it does:** Outputs the CLS embedding and Channel embeddings.
- **Channel Embedding:** The full `(B, 128, 64)` processed sequence directly usable by down-stream tasks.
- **CLS Embedding:** Global Average Pooling (GAP) across the fully processed pseudo-grid `(B, 64)`.
**Why no [CLS] Token:** The pseudo-grid approach hinges on 128 patches geometrically fitting exactly into an 8x16 matrix. A single CLS token would result in 129 patches, breaking any clean two-dimensional reshape. ViT literature heavily demonstrates that Global Average Pooling over output tokens is entirely mathematically equivalent in feature density to using a dedicated CLS token.

## Files
| File | Description |
|---|---|
| `mat_pseudo_lwm.py` | Core model: `MATPseudoLWM`, `channels_to_patches`, `mask_patches` |
| `smoke_test_pseudo.py` | 7-test smoke test + FLOPs profiler (BS=1) |
| `pretrain_pseudo.py` | Pre-training script (mirrors `lwm1_1_ca/pretraining.py`) |
| `README_pseudo.md` | This documentation |

## Quick Start
```bash
# Smoke test
conda activate lwm_cuda
python mask_aware_tf/smoke_test_pseudo.py

# Pre-training
python -m mask_aware_tf.pretrain_pseudo \
    --epochs 100 \
    --batch-size 64 \
    --save-path mask_aware_tf/mat_pseudo_lwm_weights.pth
```
