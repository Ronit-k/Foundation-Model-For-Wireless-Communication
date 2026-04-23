"""
smoke_test_pseudo.py — Quick verification that the MAT Pseudo-Grid LWM model
                       imports, runs, and profiles correctly.

Updated for the MAT-aligned architecture:
  - Convolutional stem (no Linear patch_proj / pos_embed)
  - Spatial masking on the 8×16 grid
  - Continuous mask updating after every ATB stage
  - ConvTranspose2d decoder back to (B,2,32,32)

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

# ── Test 1: Legacy helper — LWM patching still works ────────
print("\n[1] LWM patching (legacy helper) ...")
from mask_aware_tf.mat_pseudo_lwm import channels_to_patches

x = torch.randn(4, 2, 32, 32)
patches = channels_to_patches(x, patch_size=16)
assert patches.shape == (4, 128, 16), f"Got {patches.shape}"
print(f"  ✓ channels_to_patches → {tuple(patches.shape)}")

# ── Test 2: Legacy helper — Patch masking still works ────────
print("\n[2] Patch masking (MCM-style symmetric, legacy helper) ...")
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

# ── Test 3: Conv stem shape verification ─────────────────────
print("\n[3] Conv stem output shape ...")
from mask_aware_tf.mat_pseudo_lwm import MATPseudoLWM

model_tmp = MATPseudoLWM(d_model=64, gen_raw=False)
channels = torch.randn(4, 2, 32, 32)
stem_out = model_tmp.conv_stem(channels)
assert stem_out.shape == (4, 64, 8, 16), f"Got {stem_out.shape}"
print(f"  ✓ conv_stem: (B,2,32,32) → {tuple(stem_out.shape)}")

# Verify decoder round-trip shape
decoded = model_tmp.decoder(stem_out)
assert decoded.shape == (4, 2, 32, 32), f"Decoder output: {decoded.shape}"
print(f"  ✓ decoder:   (B,64,8,16) → {tuple(decoded.shape)}")

# ── Test 4: Full model training forward + backward ───────────
print("\n[4] MATPseudoLWM training forward + backward ...")
model = MATPseudoLWM(gen_raw=False, snr_db=None, mask_ratio=0.15)
n_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model built — {n_params:,} parameters")

# Verify pos_embed exists (2D spatial positional embedding)
assert hasattr(model, 'pos_embed'), "pos_embed should exist!"
assert model.pos_embed.shape == (1, 64, 8, 16), f"pos_embed shape: {model.pos_embed.shape}"
print(f"  ✓ 2D positional embedding: {tuple(model.pos_embed.shape)}")

# Verify no patch_proj exists
assert not hasattr(model, 'patch_proj'), "patch_proj should be replaced by conv_stem!"
print(f"  ✓ No linear patch_proj (using conv stem)")

loss, pred_masked, target_masked = model(channels)

assert loss.dim() == 0, f"Expected scalar loss, got {loss.shape}"
assert pred_masked.dim() == 1, f"Expected flat 1D pred, got {pred_masked.shape}"
assert pred_masked.shape == target_masked.shape, \
    f"Shape mismatch: pred {pred_masked.shape} vs target {target_masked.shape}"
assert pred_masked.numel() > 0, "No masked pixels — masking may be broken"
print(f"  ✓ loss           : {loss.item():.5f}")
print(f"  ✓ pred_masked    : {tuple(pred_masked.shape)}  (flat masked pixels)")
print(f"  ✓ target_masked  : {tuple(target_masked.shape)}")

loss.backward()
print(f"  ✓ backward() OK")

# ── Test 5: Inference mode (gen_raw=True) ────────────────────
print("\n[5] gen_raw=True (inference mode, embedding extraction) ...")
model_raw = MATPseudoLWM(gen_raw=True, d_model=64)
with torch.no_grad():
    cls_emb, channel_emb = model_raw(channels)

assert cls_emb.shape == (4, 64), f"CLS shape: {cls_emb.shape}"
assert channel_emb.shape == (4, 128, 64), f"Channel shape: {channel_emb.shape}"
print(f"  ✓ cls_embedding     : {tuple(cls_emb.shape)}")
print(f"  ✓ channel_embedding : {tuple(channel_emb.shape)}  (LWM-compatible)")

# ── Test 6: Continuous mask updating verification ────────────
print("\n[6] Continuous mask updating ...")
import torch.nn.functional as F

B_test, grid_h, grid_w, win = 2, 8, 16, 4
# Generate a mask with some zeros
vm = MATPseudoLWM._generate_spatial_mask(B_test, grid_h, grid_w, 0.15, torch.device('cpu'))
valid_before = vm.sum().item()

# Thaw once (simulates after stage 1)
vm1 = model._thaw_mask(vm, win)
valid_after_1 = vm1.sum().item()
assert valid_after_1 >= valid_before, "Thawing should not reduce valid count"

# Thaw again (simulates after stage 2)
vm2 = model._thaw_mask(vm1, win)
valid_after_2 = vm2.sum().item()
assert valid_after_2 >= valid_after_1, "Second thaw should not reduce valid count"

print(f"  ✓ Initial valid cells : {valid_before:.0f}")
print(f"  ✓ After stage1 thaw  : {valid_after_1:.0f}  (≥ initial)")
print(f"  ✓ After stage2 thaw  : {valid_after_2:.0f}  (≥ stage1)")
print(f"  ✓ Progressive mask thawing verified")

# ── Test 7: Self-containment check ───────────────────────────
print("\n[7] Self-containment check ...")
src = open(os.path.join(ROOT, "mask_aware_tf", "mat_pseudo_lwm.py")).read()
assert "mat_vit_lwm" not in src, "Still imports from mat_vit_lwm!"
print(f"  ✓ No imports from mat_vit_lwm — fully self-contained")

# ── Test 8: Package exports ──────────────────────────────────
print("\n[8] Package __init__ exports ...")
from mask_aware_tf import MATPseudoLWM  # noqa
print("  ✓ MATPseudoLWM importable from mask_aware_tf")

# ── Test 9: Architecture Summary & FLOPs Profiling ───────────
print("\n[9] Architecture Summary & FLOPs Profiling ...")
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
    print("  Run: `pip install torchinfo thop colorama` to enable Test 9.")
    print(f"  Error details: {e}")

print()
print("=" * 60)
print("  All tests PASSED ✓")
print("=" * 60)
