"""
smoke_test_pseudo.py — Quick verification that the MAT Pseudo-Grid LWM model
                       imports, runs, and profiles correctly.

Run from the project root with:
    conda activate lwm_cuda
    python mask_aware_tf/smoke_test_pseudo.py
"""
import sys
import os

# Make sure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

print("=" * 60)
print("  MAT Pseudo-Grid LWM Smoke Test & Profiler")
print("=" * 60)

# ── Test 1: LWM patching ────────────────────────────────────
print("\n[1] LWM patching ...")
from mask_aware_tf.mat_pseudo_lwm import channels_to_patches

x = torch.randn(4, 2, 32, 32)
patches = channels_to_patches(x, patch_size=16)
assert patches.shape == (4, 128, 16), f"Got {patches.shape}"
print(f"  ✓ channels_to_patches → {tuple(patches.shape)}")

# ── Test 2: Patch masking ────────────────────────────────────
print("\n[2] Patch masking (MCM-style symmetric) ...")
from mask_aware_tf.mat_pseudo_lwm import mask_patches

masked_p, mask_1d, mpos, mtokens = mask_patches(patches, mask_ratio=0.15)
n_masks = mpos.size(1)
assert masked_p.shape == patches.shape
assert mask_1d.shape == (4, 128)
assert mtokens.shape == (4, n_masks, 16)
print(f"  ✓ masked_patches : {tuple(masked_p.shape)}")
print(f"  ✓ mask_1d        : {tuple(mask_1d.shape)}  (valid={mask_1d.sum(1)[0].int().item()}/128)")
print(f"  ✓ masked_pos     : {tuple(mpos.shape)}  ({n_masks} masked patches)")
print(f"  ✓ masked_tokens  : {tuple(mtokens.shape)}")

# Verify symmetry: real and imag halves have same masked positions
half = n_masks // 2
assert torch.equal(mpos[:, :half], mpos[:, half:] - 64), "Symmetric masking broken"
print(f"  ✓ Symmetric masking verified (real↔imag)")

# ── Test 3: 2D Bridge reshape ────────────────────────────────
print("\n[3] 2D Bridge reshape ...")
B, d_model, grid_h, grid_w = 4, 64, 8, 16
seq = torch.randn(B, 128, d_model)

# 1D → 2D
grid = seq.permute(0, 2, 1).view(B, d_model, grid_h, grid_w)
assert grid.shape == (B, 64, 8, 16), f"Got {grid.shape}"
print(f"  ✓ 1D→2D: (B,128,64) → {tuple(grid.shape)}")

# 2D → 1D (round-trip)
seq_back = grid.view(B, d_model, -1).permute(0, 2, 1)
assert torch.allclose(seq, seq_back)
print(f"  ✓ 2D→1D round-trip verified")

# ── Test 4: Full model training forward + backward ───────────
print("\n[4] MATPseudoLWM training forward + backward ...")
from mask_aware_tf.mat_pseudo_lwm import MATPseudoLWM

model = MATPseudoLWM(gen_raw=False, snr_db=None, mask_ratio=0.15)
n_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model built — {n_params:,} parameters")

channels = torch.randn(4, 2, 32, 32)
loss, logits_masked, target_masked = model(channels)

assert loss.dim() == 0, f"Expected scalar loss, got {loss.shape}"
assert logits_masked.shape == target_masked.shape
assert logits_masked.shape[2] == 16, f"Decode dim: {logits_masked.shape}"
print(f"  ✓ loss           : {loss.item():.5f}")
print(f"  ✓ logits_masked  : {tuple(logits_masked.shape)}")
print(f"  ✓ target_masked  : {tuple(target_masked.shape)}")

loss.backward()
print(f"  ✓ backward() OK")

# ── Test 5: Inference mode (gen_raw=True) ────────────────────
print("\n[5] gen_raw=True (inference mode, embedding extraction) ...")
model_raw = MATPseudoLWM(gen_raw=True)
with torch.no_grad():
    cls_emb, channel_emb = model_raw(channels)

assert cls_emb.shape == (4, 64), f"CLS shape: {cls_emb.shape}"
assert channel_emb.shape == (4, 128, 64), f"Channel shape: {channel_emb.shape}"
print(f"  ✓ cls_embedding     : {tuple(cls_emb.shape)}")
print(f"  ✓ channel_embedding : {tuple(channel_emb.shape)}  (LWM-compatible)")

# ── Test 6: Package exports ──────────────────────────────────
print("\n[6] Package __init__ exports ...")
from mask_aware_tf import MATPseudoLWM  # noqa
print("  ✓ MATPseudoLWM importable from mask_aware_tf")

# ── Test 7: Architecture Summary & FLOPs Profiling ───────────
print("\n[7] Architecture Summary & FLOPs Profiling ...")
try:
    from torchinfo import summary
    from thop import profile
    import colorama
    from colorama import Fore, Style

    colorama.init(autoreset=True)

    profiling_model = MATPseudoLWM(gen_raw=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiling_model.to(device)
    dummy_input = torch.randn(1, 2, 32, 32).to(device)

    print(f"\n{Fore.YELLOW}--- Torchinfo Architecture Summary ---")
    summary(
        profiling_model,
        input_data=dummy_input,
        verbose=1,
        col_names=["input_size", "output_size", "num_params"],
        device=device,
    )

    print(f"\n{Fore.YELLOW}--- THOP FLOPs & Params Profiling ---")
    flops, params = profile(
        profiling_model,
        inputs=(dummy_input,),
        verbose=False,
    )

    print(f"{Fore.MAGENTA}- Total FLOPs (BS=1): {Fore.CYAN}{flops/1e6:.2f} MFLOPs "
          f"{Style.DIM}({flops/1e9:.4f} GFLOPs){Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}- Total Parameters:   {Fore.CYAN}{params/1e6:.3f} M{Style.RESET_ALL}")
    print("  ✓ Profiling successful.")

except ImportError as e:
    print("  [Skipping] Missing library for profiling.")
    print("  Run: `pip install torchinfo thop colorama` to enable Test 7.")
    print(f"  Error details: {e}")

print()
print("=" * 60)
print("  All tests PASSED ✓")
print("=" * 60)
