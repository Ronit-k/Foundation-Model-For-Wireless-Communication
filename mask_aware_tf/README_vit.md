# MAT-ViT-LWM: Method 1 — 32×32 Vision Transformer Approach

A hybrid foundation model combining the **Mask-Aware Transformer (MAT)** with **LWM-compatible embeddings** for wireless channel modelling. Operating strictly on native 2D grids, this resolves spatial distortion issues inherent to 1D patch flattening.

## Architecture Overview

```text
Input (B, 2, 32, 32)         ← Real + Imaginary channels
         │
    1×1 Conv2d(2 → 64)       ← Initial projection
         │
   (B, 64, 32, 32)
         │
   ┌─ Spatial Masking (15%) ┐
   │  (B, 64, 32, 32)       │  ← Random pixels zeroed
   │  + mask (B, 1, 32, 32) │  ← Validity mask generated
   └────────────────────────┘
         │
   ┌─ 3-Stage ATB Body ─────────────┐
   │ Stage 1: ATB (4x4)             │
   │   │                            │
   │ ┌─┴─ Dynamic Mask Relaxation ┐ │
   │ │ Mask MaxPool2d (Downsample)│ │ ← 8x8 MaxPool on mask to find valid pixels
   │ │ Repeat Interleave (Upsample) │ ← 8x8 Upsample back to (B,1,32,32)
   │ └─┬──────────────────────────┘ │
   │   │                            │
   │ Stage 2: ATB (8x8) Shifted     │ ← Shifted large window on relaxed mask
   │ Stage 3: ATB (4x4) Standard    │
   └────────────────────────────────┘
         │
   (B, 64, 32, 32)           ← MAT output features
         │
   ┌─────┴──────────────────────────────────────┐
   │                                            │
   1×1 Conv2d(64 → 2)                     Inference Only:
   │                                     ┌──────┴──────┐
   (B, 2, 32, 32)                   GAP→(B,64)    Pool→(B,128,64)
   │                              CLS embedding   Channel embedding
   MSE Loss (masked pixels only)
```

## Detailed Layer-by-Layer Architecture & Design Choices

### 1. Initial Spatial Projection (`proj_in`)
**What it does:** Receives raw continuous complex wireless data `(B, 2, 32, 32)`. A 1x1 Convolution maps the two channels directly to an embedding dimension `d_model` (e.g., 64), yielding `(B, 64, 32, 32)`. 
**Why this choice:** Instead of breaking the radio map into flat patches (which destroys 2D vertical contiguity), a 1x1 Conv pointwise linear projection promotes all spatial coordinates efficiently into high-dimensional feature space while strictly maintaining the native 32x32 topology.

### 2. Spatial Pixel Masking
**What it does:** Generates a random binary mask covering exactly 15% of the `32x32` spatial map (`generate_spatial_mask`). The mask zeros out randomly selected individual pixels in feature space before passing it to the transformer.
**Why this choice:** Replaces standard ViT patch-dropping with fine-grained pixel-level masking. Missing pixel interpolation is closely analogous to spatial channel estimation issues faced inherently in multiple input multiple output (MIMO) fading matrices, perfectly modeling realistic wireless communication environments.

### 3. 3-Stage Adjusted Transformer Block (ATB) Body
**What it does:** A 3-stage attention pipeline utilizing Mask-Aware Multi-Head Attention.
**Layer breakdown within each ATB:**
- **WindowMHA (Mask-Aware Attention):** MHA running solely in localized spatial windows. Masked, invalid tokens are injected with a massive negative bias (`-tau` usually 100) right before the softmax, forcing them to contribute exactly 0 to surrounding unmasked pixels.
- **Concat + 1×1 Conv Fusion:** The WindowMHA result is channel-concatenated back with the untouched input and down-projected using a `1x1` convolution.
- **MLP & 3x3 Local Conv:** Operates parallel paths: A standard two-layer network with GELU, alongside a `3x3` convolution with padding=1.
**Why these choices:** 
1. **Mask-Aware Window Attn:** Softmax automatically averages values over a denominator. If masked tokens equal zero but are included in the denominator, they systematically dull the attention score (an effect known as "leakage"). The mask-aware mechanism fully nullifies invalid pixel leakage.
2. **Local Conv:** Window attention causes heavy boundary grid artifacts since it never looks outside its window block. Parallel `3x3` padding-invariant convolutions guarantee local smoothing seamlessly across window boundaries.
3. **No LayerNorm:** Standardization involves mean calculation over entire sequences. A few empty (masked) pixels would inherently shift the global mean and standard deviation, leaking their absence globally. Eliminating LayerNorm enforces strict causal validity. 

### 4. Multi-Scale Window Strategy & Dynamic Updating
**What it does:** Stage 1 uses `4x4` windows. Stage 2 uses `8x8` shifted windows. Stage 3 returns to `4x4` windows. Following Stage 1, a 2D MaxPool relaxes the mask—meaning if any single pixel in an `8x8` block was valid, all tokens in the `8x8` block are now conceptually marked as "valid" for Stage 2.
**Why this choice:** Inspired by U-Net and Hourglass philosophy. Stage 1 conducts hyper-localized (4x4) initial inference. By dynamically marking guessed tokens as valid and expanding the receptive field to shifted 8x8 blocks in Stage 2, it propagates synthesized confidence over longer boundaries. Stage 3 handles local high-frequency refinement.

### 5. Decoding and MSE Validation (`proj_out`)
**What it does:** A 1x1 Convolution condenses `(B, 64, 32, 32)` down to the predicted `(B, 2, 32, 32)`. An MSE loss computes error strictly on the indices where pixels were randomly masked.
**Why MSE over Perceptual GAN Loss:** The original MAT paper (designed for image inpainting) relies on GAN-based VGG perceptual loss. Wireless signals are physics-governed continuous float maps, not semantic human imagery. VGG features are practically useless on frequency spectrums. Mean Squared Error (MSE) acts exactly as Expected Signal Fidelity, the only metric mathematically correlated with downstream data rate.

### 6. Inference and LWM Compatibility
**What it does:** Extracts two embeddings during `gen_raw=True`.
- **CLS Embedding:** Global average pooling (GAP).
- **Channel Embedding:** Standard 2D Adaptive Average Pooling `AdaptiveAvgPool2d((8, 16))` shrinks the 32x32 spatial map to 8x16, then reshapes it to sequence format `(B, 128, 64)`.
**Why this choice:** Adaptive Pooling mathematically bridges the spatial gap. It forcefully down-samples the highly detailed 32x32 maps natively to exactly 128 tokens of dimension 64. By exposing this array, the model behaves as a drop-in 1-to-1 replacement for LWM's dense embedding backbone, completely omitting the need to update any LWM orchestration code. 


## Files
| File | Description |
|---|---|
| `mat_vit_lwm.py` | Core model: `WindowMHA`, `ATB`, `MATViTLWM` |
| `smoke_test_vit.py` | Smoke test + FLOPs profiler (BS=1) |
| `pretrain_vit.py` | Pre-training script (mirrors `lwm1_1_ca/pretraining.py`) |
| `README_vit.md` | This documentation |

## Quick Start
```bash
# Smoke test
conda activate lwm_cuda
python mask_aware_tf/smoke_test_vit.py

# Pre-training
python -m mask_aware_tf.pretrain_vit \
    --epochs 100 \
    --batch-size 64 \
    --save-path mask_aware_tf/mat_vit_lwm_weights.pth
```
