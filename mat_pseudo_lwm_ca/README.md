# MAT Pseudo-Grid LWM + Coordinate Attention (CA)

A hybrid foundation model that combines **MAT's mask-aware 2D spatial attention** with a **Coordinate Attention front-end** for wireless channel pretraining. Extends the base MAT Pseudo-Grid LWM (`mask_aware_tf/mat_pseudo_lwm.py`) by inserting a CoordAtt module that recalibrates the raw channel along both antenna (H) and subcarrier (W) axes before tokenization.

## Architecture Overview

```text
Input (B, 2, 32, 32)         ← Real + Imaginary channels
         │
   ┌─────┴──────────┐
   │  CoordAtt      │        ← Coordinate Attention (H=antennas, W=subcarriers)
   │  (2→2 ch)      │          Learns H-direction and W-direction attention maps
   │  identity×a_h×a_w │       separately via directional pooling
   └─────┬──────────┘
         │
   (B, 2, 32, 32)            ← Spatially recalibrated channels
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
   │ │ Mask MaxPool2d (Downsample)│ │ ← Find any valid tokens in window
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

### 0. Coordinate Attention Front-End (`CoordAtt`)
**What it does:** Before any tokenisation, the raw `(B, 2, 32, 32)` channel (real + imaginary) passes through a Coordinate Attention block. CA pools the feature map separately along the H (antenna) and W (subcarrier) axes, concatenates the two directional contexts, passes them through a shared 1×1 conv bottleneck (BN + HSwish), then generates separate H-direction and W-direction sigmoid attention maps. The output is `identity × a_h × a_w`.

**Why this choice:** Unlike SENet (which uses global average pooling and loses all positional info), CoordAtt retains directional and positional context. This is highly suited for OFDM channel matrices where the antenna axis and subcarrier axis encode fundamentally different physical information (spatial vs. frequency). Training CA jointly with the ATB backbone via the MCM loss lets the attention weights adapt to emphasise channel features most informative for masked-token reconstruction.

**Parameters:** `in_channels=2, out_channels=2, reduction=32` → `mid_channels = max(8, 2//32) = 8`. Adds ~200 trainable parameters.

### 1. Input Patching (`channels_to_patches`)
**What it does:** Takes `(B, 2, 32, 32)` → flattens spatial dims → concatenates real+imag → slices into 128 non-overlapping patches of length 16 → `(B, 128, 16)`.

**Why this choice:** Mirrors the original LWM tokenisation exactly, ensuring 100% downstream compatibility with pre-existing LWM infrastructure.

### 2. Symmetric MCM Masking (`mask_patches`)
**What it does:** Randomly drops 15% of patches (zeroed out), applying masks symmetrically — if patch `i` in the real half is masked, patch `i + 64` in the imag half is also masked.

**Why this choice:** The complex structure (Real + Imag) is physics-governed. Symmetric masking forces the network to reconstruct complete complex tokens from context, preserving the physical coupling.

### 3. Patch Embedding & Positional Encoding
**What it does:** `nn.Linear(16→64)` + learnable positional embedding → `(B, 128, 64)`.

**Why this choice:** Learnable embeddings (over trigonometric) because wireless components don't always behave as contiguous visual regions.

### 4. 2D Spatial Bridge
**What it does:** Reshapes `(B, 128, 64)` → `(B, 64, 8, 16)` pseudo-grid for 2D windowed attention.

**Why this choice:** Maps 128 tokens into an 8×16 grid, enabling MAT's 2D shifted-window attention without requiring global self-attention.

### 5. 3-Stage ATB Body
Each ATB contains: WindowMHA (mask-aware, −100 penalty for masked tokens) → Concat + 1×1 Conv fusion → MLP (GELU) with residual → 3×3 local conv with residual. No LayerNorm (prevents mask information leakage). Stage 2 uses shifted windows for cross-window information flow.

### 6. Reverse Bridge & Selective Decoder
Reshapes back to `(B, 128, 64)`, gathers only masked embeddings via `torch.gather`, decodes with `nn.Linear(64→16)`. MSE loss on masked patches only.

### 7. Inference (`gen_raw=True`)
- **CLS embedding:** GAP across pseudo-grid → `(B, 64)`
- **Channel embedding:** Full sequence → `(B, 128, 64)` (LWM-compatible)

## Files

| File | Description |
|---|---|
| `coordatt.py` | Coordinate Attention module (`CoordAtt`, `HSigmoid`, `HSwish`) |
| `mat_pseudo_lwm_ca.py` | Core model: `MATPseudoLWMWithCA`, `channels_to_patches`, `mask_patches` |
| `smoke_test.py` | 8-test smoke test + CA gradient verification + FLOPs profiler |
| `pretrain_pseudo_ca.py` | Pre-training script (mirrors `mask_aware_tf/pretrain_pseudo.py`) |
| `__init__.py` | Package exports |
| `README.md` | This documentation |

## Quick Start

```bash
# Smoke test
conda activate lwm_cuda
python mat_pseudo_lwm_ca/smoke_test.py

# Pre-training (default naming: {batch_size}_{epochs}mat_pseudo_lwm_ca_weights.pth/.log)
python -m mat_pseudo_lwm_ca.pretrain_pseudo_ca \
    --epochs 100 \
    --batch-size 512

# Explicit save path
python -m mat_pseudo_lwm_ca.pretrain_pseudo_ca \
    --epochs 100 \
    --batch-size 512 \
    --save-path mat_pseudo_lwm_ca/512_100mat_pseudo_lwm_ca_weights.pth

# With channel caching (faster repeated runs)
python -m mat_pseudo_lwm_ca.pretrain_pseudo_ca \
    --channels-cache /tmp/mat_pseudo_ca_channels.npy

# CPU smoke run (single epoch)
python -m mat_pseudo_lwm_ca.pretrain_pseudo_ca \
    --scenarios city_0_newyork \
    --epochs 1 \
    --batch-size 8 \
    --device cpu \
    --save-path /tmp/test_weights.pth
```

## Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 64 (default) |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR scheduler | StepLR |
| Step size | 10 epochs |
| Gamma | 0.9 |
| Train/Val split | 80% / 20% |
| Patch format | LWM flat, patch_size=16 |
| MCM mask ratio | 15% symmetric |
| CA reduction | 32 |
| Loss | MSE on masked patches |
| Seed | 0 |

## Output Files

After pretraining with default settings (`--batch-size 512 --epochs 100`):

| File | Description |
|---|---|
| `512_100mat_pseudo_lwm_ca_weights.pth` | Full model state dict (CA + ATB backbone) |
| `512_100mat_pseudo_lwm_ca_weights.log` | Training log with loss, LR, GPU memory per epoch |

## How It Compares to `mask_aware_tf` (Base)

| Aspect | `mask_aware_tf` (MATPseudoLWM) | `mat_pseudo_lwm_ca` (MATPseudoLWMWithCA) |
|---|---|---|
| CA front-end | ❌ None | ✅ CoordAtt(2→2) before patching |
| Backbone | 3-stage ATB (8×16 grid) | 3-stage ATB (identical) |
| Patching | LWM 128 patches × 16 | LWM 128 patches × 16 (identical) |
| Masking | 15% symmetric MCM | 15% symmetric MCM (identical) |
| Loss | MSE on masked patches | MSE on masked patches (identical) |
| Extra parameters | — | ~200 (CoordAtt bottleneck) |
| Inference output | (B,64) CLS + (B,128,64) channel | Identical shapes |
