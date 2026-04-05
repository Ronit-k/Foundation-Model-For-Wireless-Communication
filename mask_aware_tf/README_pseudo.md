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
   ┌─ 3-Stage ATB Body ──┐
   │  Stage 1: ATB (4×4)  │   ← Mask-aware + validity update
   │  Stage 2: ATB (4×4)  │   ← Shifted windows
   │  Stage 3: ATB (4×4)  │   ← Standard windows
   └──────────────────────┘
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
- Shifted windows (offset by 2) create 8 new cross-boundary windows
- 16 tokens per window is computationally efficient

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
