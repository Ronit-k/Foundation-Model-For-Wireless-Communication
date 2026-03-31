"""
smoke_test_vit.py — Quick verification that the MAT-ViT-LWM model imports,
                    runs, and profiles correctly.

Run from the project root with:
    conda activate lwm_cuda
    python mask_aware_tf/smoke_test_vit.py
"""
import sys
import os

# Make sure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

print("=" * 60)
print("  MAT-ViT-LWM Smoke Test & Profiler")
print("=" * 60)

# ── Test 1: WindowMHA ────────────────────────────────────────
print("\n[1] WindowMHA ...")
from mask_aware_tf.mat_vit_lwm import WindowMHA

mha = WindowMHA(dim=64, heads=4, win=4)
x = torch.randn(2, 64, 32, 32)
vm = torch.ones(2, 1, 32, 32)
y = mha(x, vm, shift=False)
assert y.shape == x.shape, f"shape mismatch: {y.shape}"
print(f"  ✓ WindowMHA (no shift) → {tuple(y.shape)}")

y_shift = mha(x, vm, shift=True)
assert y_shift.shape == x.shape
print(f"  ✓ WindowMHA (shifted)  → {tuple(y_shift.shape)}")

# ── Test 2: ATB ──────────────────────────────────────────────
print("\n[2] ATB (Adjusted Transformer Block) ...")
from mask_aware_tf.mat_vit_lwm import ATB

atb = ATB(dim=64, heads=4, win=4)
x = torch.randn(2, 64, 32, 32)
vm = torch.ones(2, 1, 32, 32)
y = atb(x, vm, shift=False)
assert y.shape == x.shape, f"shape mismatch: {y.shape}"
print(f"  ✓ ATB forward → {tuple(y.shape)}")

# ── Test 3: Full model training forward + backward ───────────
print("\n[3] MATViTLWM training forward + backward ...")
from mask_aware_tf.mat_vit_lwm import MATViTLWM

model = MATViTLWM(gen_raw=False, snr_db=None, mask_ratio=0.15)
n_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model built — {n_params:,} parameters")

channels = torch.randn(4, 2, 32, 32)
loss, pred_masked, target_masked = model(channels)

assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
assert pred_masked.shape == target_masked.shape, \
    f"Pred/target shape mismatch: {pred_masked.shape} vs {target_masked.shape}"
print(f"  ✓ loss          : {loss.item():.5f}")
print(f"  ✓ pred_masked   : {tuple(pred_masked.shape)}")
print(f"  ✓ target_masked : {tuple(target_masked.shape)}")

loss.backward()
print(f"  ✓ backward() OK")

# Verify expected number of masked pixels (15% of 2×32×32 = 307.2 ≈ 307-308)
n_masked = pred_masked.numel()
expected = int(0.15 * 2 * 32 * 32) * 4  # ×4 for batch
print(f"  ✓ masked pixels : {n_masked} (expected ≈ {expected})")

# ── Test 4: Inference mode (gen_raw=True) ────────────────────
print("\n[4] gen_raw=True (inference mode, embedding extraction) ...")
model_raw = MATViTLWM(gen_raw=True)
with torch.no_grad():
    cls_emb, channel_emb = model_raw(channels)

assert cls_emb.shape == (4, 64), f"CLS embedding shape: {cls_emb.shape}"
assert channel_emb.shape == (4, 128, 64), f"Channel embedding shape: {channel_emb.shape}"
print(f"  ✓ cls_embedding     : {tuple(cls_emb.shape)}")
print(f"  ✓ channel_embedding : {tuple(channel_emb.shape)}  (LWM-compatible)")

# ── Test 5: Package __init__ exports ─────────────────────────
print("\n[5] Package __init__ exports ...")
from mask_aware_tf import MATViTLWM, ATB, WindowMHA, generate_spatial_mask  # noqa
print("  ✓ MATViTLWM, ATB, WindowMHA, generate_spatial_mask importable")

# ── Test 6: Architecture Summary & FLOPs Profiling ───────────
print("\n[6] Architecture Summary & FLOPs Profiling ...")
try:
    from torchinfo import summary
    from thop import profile
    import colorama
    from colorama import Fore, Style

    colorama.init(autoreset=True)

    # Initialize model in strict inference mode
    profiling_model = MATViTLWM(gen_raw=True).eval()

    # CRITICAL: Batch Size = 1 for industry standard inference metric
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
    print("  Run: `pip install torchinfo thop colorama` to enable Test 6.")
    print(f"  Error details: {e}")

print()
print("=" * 60)
print("  All tests PASSED ✓")
print("=" * 60)
