# MAT-ViT-LWM: Method 1 — 32×32 Vision Transformer Approach

A hybrid foundation model combining the **Mask-Aware Transformer (MAT)** with
**LWM-compatible embeddings** for wireless channel modelling.

## Architecture Overview

```
Input (B, 2, 32, 32)         ← Real + Imaginary channels
         │
    1×1 Conv2d(2 → 64)       ← Initial projection
         │
   (B, 64, 32, 32)
         │
   ┌─ Spatial Masking ─┐     ← 15% random pixels zeroed
   │  (B, 1, 32, 32)   │
   └────────────────────┘
         │
   ┌─ 3 MAT Stages Body ─┐
   │  Stage 1 (3 ATB+Conv) │    ← WindowMHA + mask validity update
   │  Stage 2 (3 ATB+Conv) │    ← Middle ATB has shifted windows
   │  Stage 3 (3 ATB+Conv) │    ← Standard windows
   └───────────────────────┘
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

## 3-Stage ATB Structure

Each **Adjusted Transformer Block (ATB)** contains:

1. **WindowMHA** — Multi-Head Contextual Attention with non-overlapping windows.
   Invalid (masked) tokens receive a **-100 penalty** bias, effectively
   suppressing them in the softmax. Stage 2 uses shifted windows for cross-window
   information flow.

2. **Concat + FC** — The attention output is concatenated with the input along
   the channel dimension, then fused via a 1×1 convolution.

3. **MLP** — A two-layer feedforward network (1×1 Conv → GELU → 1×1 Conv) with
   a residual connection.

4. **Local Conv** — A 3×3 convolution with residual connection to capture local
   spatial patterns the windowed attention may miss.

### Multi-Scale Window Strategy

| Stage | Window Size | Receptive Field | Purpose |
|---|---|---|---|
| 1 | **4×4** | Local (16 tokens) | Fine-grained local reconstruction |
| 2 | **8×8** (shifted) | Wide (64 tokens) | Long-range dependency capture |
| 3 | **4×4** | Local (16 tokens) | Refinement with local context |

The 4→8→4 pattern is inspired by the hourglass/U-Net philosophy: start with
local attention for initial reconstruction, expand to capture long-range
correlations, then refine locally.

**No LayerNorm** is used in the ATBs. This follows the MAT paper's design for
inpainting, where normalisation layers can leak mask information.

### Dynamic Mask Updating

After Stage 1, the validity mask is relaxed via max-pooling over Stage 2's
8×8 window regions: if **any** token within an 8×8 window is valid, **all**
tokens in that window become valid. This allows the larger Stage 2 windows
to attend to reconstructed features from Stage 1.

## Deeper Architecture: MATStage

To match the 12-layer depth of the original Large Wireless Model (LWM), the model utilizes 3 full `MATStage` modules. Each `MATStage` contains:

1. 3 sequential ATBs (yielding 9 ATBs total).
2. The middle ATB of each stage uses shifted windows (`shift=True`) for cross-window communication.
3. A `3x3` localized convolutional layer wrapping the block.
4. A macro skip-connection connecting the start and end of the stage.

This gives the model the parameter depth required to rival the original LWM while retaining MAT's 2D dynamic noise-filtering.

## Why MSE Loss (Not GAN/VGG Perceptual)

The original MAT uses adversarial + perceptual losses because it generates
natural images where human perception matters. Wireless channel data is
fundamentally different:

| Property | Natural Images | Wireless Channels |
|---|---|---|
| **Data type** | Discrete RGB pixels | Continuous complex values |
| **Quality metric** | Human perception (SSIM, LPIPS) | Signal fidelity (MSE, NMSE) |
| **Distribution** | Multi-modal, highly structured | Smooth, physics-governed |
| **Downstream use** | Visual tasks | Beam prediction, positioning |

**MSE is the standard loss for wireless channel estimation** because:
- Channel data has a well-defined ground truth (not perceptually ambiguous)
- Downstream tasks (beam prediction, rate estimation) depend on signal accuracy
- GAN training introduces instability without perceptual benefit
- VGG features are trained on ImageNet — meaningless for wireless spectra

This matches the loss used in LWM v1.0 and v1.1 pretraining.

## Embedding Extraction (Inference)

When `gen_raw=True`, the model produces two outputs:

- **CLS Embedding** `(B, 64)` — Global Average Pooling over the full spatial
  output, representing the entire channel matrix in a single vector
- **Channel Embedding** `(B, 128, 64)` — `AdaptiveAvgPool2d((8, 16))` followed
  by flatten+transpose, producing a 128-token sequence of dimension 64 that is
  directly compatible with LWM's downstream heads

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
