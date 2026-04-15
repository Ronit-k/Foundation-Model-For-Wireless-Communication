# MAT Pseudo-Grid LWM: Method 2 — 8×16 Pseudo-Grid Approach

A hybrid foundation model that preserves **LWM's 128-patch tokenization** while
enabling **MAT's 2D mask-aware spatial attention** via an 8×16 pseudo-grid reshape.

## Architecture Overview

```
Input (B, 2, 32, 32)         ← Real + Imaginary channels
         │
   LWM Patching              ← Flatten + slice into 128 patches of length 16
   (B, 128, 16)
         │
   MCM Masking (15%)         ← Symmetric: same positions in real & imag halves
   (B, 128, 16) + mask (B,128)
         │
   Linear(16→64) + PosEmb    ← Patch embedding + learnable positional encoding
   (B, 128, 64)
         │
   ┌─ 2D Bridge ─────────┐
   │ reshape (B,128,64)   │   ← Permute + view into pseudo-grid
   │   → (B,64,8,16)     │
   │ mask → (B,1,8,16)   │
   └──────────────────────┘
         │
   ┌─ 3 MAT Stages Body (LayerScaled) ─┐
   │  Stage 1 (2 ATB+Conv)          │   ← Mask-aware + LayerScale (γ=1e-2)
   │  Stage 2 (2 ATB+Conv)          │   ← Shifted windows + LayerScale
   │  Stage 3 (2 ATB+Conv)          │   ← Standard windows + LayerScale
   └───────────────────────────────────┘
         │
   Reverse 2D Bridge         ← (B,64,8,16) → (B,128,64)
   (B, 128, 64)
         │
   ┌─────┴──────────────────────────────────────┐
   │                                            │
   torch.gather(masked_pos)              Inference Only:
   → (B, n_masks, 64)                  ┌──────┴──────┐
   → Linear(64→16)                  GAP→(B,64)    (B,128,64)
   → (B, n_masks, 16)            CLS embedding  Channel emb
   │
   MSE Loss vs raw patches
```

## Why No [CLS] Token

The CLS token must be **dropped** because:

1. **Geometric constraint**: 128 patches = 8 × 16 exactly. Adding a CLS token
   (129 patches) would break the clean rectangular reshape.
2. **No padding needed**: Without CLS, every position in the 8×16 grid maps
   directly to a real patch — no wasted computation.
3. **GAP replacement**: Global average pooling over the full pseudo-grid produces
   equal or better classification-quality embeddings than a single CLS token
   (demonstrated in ViT literature, e.g., "An Image is Worth 16x16 Words").

## Why 4×4 Windows

The 8×16 pseudo-grid is small — only 128 spatial positions. Using large windows
would reduce the number of windows and hurt the shifted-window mechanism:

| Window | Grid: 8×16 | Windows | Tokens/Win | Viable? |
|---|---|---|---|---|
| 16×16 | ✗ | 8 not divisible by 16 | — | **No** |
| 8×8 | ✗ | 1×2 = 2 windows | 64 | Borderline |
| **4×4** | ✓ | 2×4 = **8 windows** | 16 | **Optimal** |

With 4×4 windows:
- 8 non-overlapping windows tile the 8×16 grid perfectly
- Shifted windows (offset by 2) within `MATStage` create 8 new cross-boundary windows
- 16 tokens per window is computationally efficient

## Deeper Architecture: MATStage & The LayerScale Imperative
To rival the depth of the original 12-layer LWM, the model utilizes 3 full `MATStage` modules. Each `MATStage` evaluates:
1. 2 sequential ATBs (yielding 6 ATBs total across the network).
2. The second ATB of each stage utilizes shifted windows (`shift=True`) for cross-window communication.
3. A `3x3` localized convolutional layer mapping wrapping the final ATB block.
4. A macroscopic residual logic connecting the identity wrapper to the end feature transformation.

### The No-LayerNorm LayerScale Stabilization Protocol
In standard deep architectures (ViTs/ResNets), data passing heavily through 9 sequential Transformer blocks would accumulate cascading variance, leading rapidly to catastrophic gradient explosion (NaNs/Loss in the millions). This is classically countered using `LayerNorm` buffers.
However, **normalization layers strictly average statistical data over the token sequence**. In a Masked Image Modeling regime, averaging visible tokens inherently mathematically leaks the position and relative absence of the missing (`0`) tokens—destroying the entire self-supervised reconstruction objective.

To counter this, **LayerScale** is strictly applied across the entire network body:
- **Zero Leakage:** Instead of averaging tokens across the spatial grid, each branch connection (`mlp`, `local_conv`, and `stage_conv`) multiplied exclusively by a fixed, uniquely learnable scalar parameter `gamma`. Token A computes itself mathematically completely blind to the presence of Token B. 
- **Exploding Gradient Nullification:** Every `gamma` is initialized to `1e-2` scale. This aggressively chokes the starting residual contribution, essentially compelling the deep 6-layer `Heavy_MAT` model to strictly feign a completely safe and shallow sub-network at Epoch 0, while still allowing gradients to flow deep into the network. As convergence aligns smoothly over descending epochs, the model gently trains the `gamma` scales up to introduce deeper representation fidelity.
 
Despite increasing from 3 to 6 total Transformer blocks, the 16-length patch tokenization keeps the FLOP count highly efficient.

## Decoding & Loss

Unlike Method 1 (which computes pixel-level MSE), Method 2 uses **patch-level**
masked channel modelling:

1. After the ATB body, reshape back to `(B, 128, 64)` sequence
2. Use `torch.gather` to extract **only** the masked patch embeddings → `(B, n_masks, 64)`
3. Project through `nn.Linear(64, 16)` → `(B, n_masks, 16)`
4. Compute `F.mse_loss(predicted_patches, original_patches)`

This is directly compatible with LWM's MCM pre-training objective.

### Symmetric Masking

Masking is applied symmetrically: if patch `i` in the real half (0–63) is masked,
patch `i+64` in the imaginary half is also masked. This preserves the complex
structure of wireless channel data during reconstruction.

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
