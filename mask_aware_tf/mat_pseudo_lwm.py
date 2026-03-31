# =============================================================================
# mat_pseudo_lwm.py — MAT Pseudo-Grid LWM (Method 2: 8×16 Pseudo-Grid)
#
# Preserves LWM's 128-patch tokenization while enabling MAT's 2D spatial
# attention via an 8×16 pseudo-grid reshape.
#
# Data flow:
#   (B,2,32,32) → flatten → (B,128,16) → mask 15% → Linear(16→64) + PosEmb
#   → (B,128,64) → reshape → (B,64,8,16) → 3-stage ATB → reshape back
#   → (B,128,64) → gather masked → Linear(64→16) → MSE vs raw patches
#
# Inference:
#   - CLS embedding:     GAP on pseudo-grid → (B, 64)
#   - Channel embedding: flattened sequence  → (B, 128, 64)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse ATB and WindowMHA from Method 1
from .mat_vit_lwm import ATB, WindowMHA


# =============================================================================
# Helpers — LWM-compatible patching
# =============================================================================

def channels_to_patches(channels_ri: torch.Tensor,
                        patch_size: int = 16) -> torch.Tensor:
    """
    Convert (B, 2, H, W) into patches (B, n_patches, patch_size).

    Mirrors the LWM patching: flatten each channel, concatenate real and imag,
    then slice into non-overlapping patches of length ``patch_size``.
    """
    batch_size, _, height, width = channels_ri.shape
    flat = channels_ri.view(batch_size, 2, height * width)   # (B, 2, 1024)
    flat = torch.cat([flat[:, 0], flat[:, 1]], dim=1)        # (B, 2048)
    if flat.size(1) % patch_size != 0:
        raise ValueError("Flattened length not divisible by patch_size.")
    return flat.view(batch_size, -1, patch_size)             # (B, 128, 16)


def mask_patches(patches: torch.Tensor,
                 mask_ratio: float = 0.15) -> tuple:
    """
    Mask a fraction of patches using MCM-style symmetric masking.

    For LWM compatibility, masks are applied symmetrically: the same positions
    in the real half (first 64 patches) and imaginary half (last 64 patches)
    are masked together.

    Args:
        patches:    (B, 128, 16) input patches.
        mask_ratio: Fraction of real-half patches to mask.

    Returns:
        masked_patches: (B, 128, 16) with masked patches zeroed out.
        mask:           (B, 128) binary mask (1=valid, 0=masked).
        masked_pos:     (B, n_masks) indices of masked positions.
        masked_tokens:  (B, n_masks, 16) original values at masked positions.
    """
    batch_size, n_patches, patch_size = patches.shape
    real_tokens = n_patches // 2                              # 64
    n_masks_half = max(1, int(mask_ratio * real_tokens))

    # Symmetric masking: same positions in real and imaginary halves
    rand = torch.rand(batch_size, real_tokens, device=patches.device)
    pos_real = rand.topk(n_masks_half, dim=1).indices         # (B, n_masks_half)
    pos_imag = pos_real + real_tokens                          # mirror in imag half
    masked_pos = torch.cat([pos_real, pos_imag], dim=1)       # (B, n_masks)
    n_masks = masked_pos.size(1)

    # Extract original tokens at masked positions
    masked_tokens = torch.gather(
        patches, 1,
        masked_pos.unsqueeze(-1).expand(-1, -1, patch_size)
    ).detach()                                                # (B, n_masks, 16)

    # Build binary mask (1=valid, 0=masked)
    mask = torch.ones(batch_size, n_patches, device=patches.device)
    batch_idx = torch.arange(batch_size, device=patches.device)[:, None]
    mask[batch_idx.expand_as(masked_pos), masked_pos] = 0.0

    # Zero out masked patches
    mask_expand = mask.unsqueeze(-1)                          # (B, 128, 1)
    masked_patches = patches * mask_expand                    # (B, 128, 16)

    return masked_patches, mask, masked_pos, masked_tokens


# =============================================================================
# MAT Pseudo-Grid LWM Model
# =============================================================================

class MATPseudoLWM(nn.Module):
    """
    Hybrid MAT–LWM model using the 8×16 pseudo-grid approach (Method 2).

    Preserves LWM's 128-patch tokenization (patches of length 16) and reshapes
    the 1D sequence into a 2D pseudo-grid for mask-aware spatial attention.

    Training mode (``gen_raw=False``):
        Tokenize → mask → embed → 2D reshape → ATB → 1D reshape → decode
        → MSE loss on masked patches.

    Inference mode (``gen_raw=True``):
        Tokenize → embed → 2D reshape → ATB → embeddings.

    Args:
        patch_size:  Length of each patch.  Default 16.
        d_model:     Embedding dimension.   Default 64.
        n_patches:   Number of patches.     Default 128 (= 8 × 16 grid).
        grid_h:      Pseudo-grid height.    Default 8.
        grid_w:      Pseudo-grid width.     Default 16.
        heads:       Attention heads.       Default 4.
        win:         Window size for ATBs.  Default 4.
        mask_ratio:  Masking fraction.      Default 0.15.
        gen_raw:     Inference mode flag.   Default False.
        snr_db:      Optional AWGN noise.   Default None.
    """

    def __init__(
        self,
        patch_size: int = 16,
        d_model: int = 64,
        n_patches: int = 128,
        grid_h: int = 8,
        grid_w: int = 16,
        heads: int = 4,
        win: int = 4,
        mask_ratio: float = 0.15,
        gen_raw: bool = False,
        snr_db: float | None = None,
    ):
        super().__init__()
        assert grid_h * grid_w == n_patches, \
            f"Grid {grid_h}×{grid_w} != {n_patches} patches"

        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = n_patches
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.mask_ratio = mask_ratio
        self.gen_raw = gen_raw
        self.snr_db = snr_db

        # ── Patch embedding: 16 → 64 ────────────────────────────────────
        self.patch_proj = nn.Linear(patch_size, d_model)

        # ── Positional encoding (learnable) ──────────────────────────────
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches, d_model) * 0.02
        )

        # ── 3-stage ATB body (4×4 windows on 8×16 grid) ─────────────────
        self.stage1 = ATB(d_model, heads=heads, win=win)
        self.stage2 = ATB(d_model, heads=heads, win=win)
        self.stage3 = ATB(d_model, heads=heads, win=win)

        # ── Decoder: project masked embeddings back to patch space ───────
        self.decoder = nn.Linear(d_model, patch_size)

    # ---- Noise injection ------------------------------------------------
    @staticmethod
    def _add_complex_noise_ri(channels_ri: torch.Tensor,
                              snr_db: float) -> torch.Tensor:
        """Add complex Gaussian noise to (B, 2, H, W) real/imag channels."""
        real = channels_ri[:, 0]
        imag = channels_ri[:, 1]
        power = (real ** 2 + imag ** 2).mean(dim=(1, 2), keepdim=True)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = power / snr_linear
        noise_std = torch.sqrt(noise_power / 2)

        out = channels_ri.clone()
        out[:, 0] = real + torch.randn_like(real) * noise_std
        out[:, 1] = imag + torch.randn_like(imag) * noise_std
        return out

    # ---- Forward --------------------------------------------------------
    def forward(self, channels: torch.Tensor):
        """
        Args:
            channels: (B, 2, 32, 32) real/imag wireless channel data.

        Returns (training, gen_raw=False):
            loss:           Scalar MSE loss on masked patches.
            logits_masked:  (B, n_masks, 16) predicted masked patches.
            target_masked:  (B, n_masks, 16) ground-truth masked patches.

        Returns (inference, gen_raw=True):
            cls_embedding:     (B, 64) global average pooled.
            channel_embedding: (B, 128, 64) full sequence embeddings.
        """
        B = channels.size(0)

        # Optional noise augmentation
        if self.snr_db is not None:
            channels = self._add_complex_noise_ri(channels, self.snr_db)

        # ── Step 1: LWM patching ─────────────────────────────────────────
        patches = channels_to_patches(channels, self.patch_size)  # (B, 128, 16)

        # ── Inference mode ───────────────────────────────────────────────
        if self.gen_raw:
            # Embed all patches (no masking)
            x = self.patch_proj(patches) + self.pos_embed      # (B, 128, 64)

            # 2D Bridge: (B, 128, 64) → (B, 64, 8, 16)
            x_2d = x.permute(0, 2, 1).view(B, self.d_model, self.grid_h, self.grid_w)
            vm = torch.ones(B, 1, self.grid_h, self.grid_w,
                            device=x.device, dtype=x.dtype)

            # 3-stage ATB body
            x_2d = self.stage1(x_2d, vm, shift=False)
            x_2d = self.stage2(x_2d, vm, shift=True)
            x_2d = self.stage3(x_2d, vm, shift=False)

            # CLS embedding: GAP → (B, 64)
            cls_emb = x_2d.mean(dim=(2, 3))

            # Channel embedding: (B, 64, 8, 16) → (B, 128, 64)
            channel_emb = x_2d.view(B, self.d_model, -1).permute(0, 2, 1)

            return cls_emb, channel_emb

        # ── Training mode ────────────────────────────────────────────────
        # Step 2: Mask 15% of patches
        masked_patches, mask_1d, masked_pos, masked_tokens = mask_patches(
            patches, mask_ratio=self.mask_ratio
        )
        # masked_patches: (B, 128, 16) with masked positions zeroed
        # mask_1d:        (B, 128)     binary mask
        # masked_pos:     (B, n_masks) indices
        # masked_tokens:  (B, n_masks, 16) ground truth

        # Step 3: Linear embedding + positional encoding
        x = self.patch_proj(masked_patches) + self.pos_embed   # (B, 128, 64)

        # Step 4: 2D Bridge — reshape to pseudo-grid
        x_2d = x.permute(0, 2, 1).view(B, self.d_model, self.grid_h, self.grid_w)
        # Reshape mask: (B, 128) → (B, 1, 8, 16)
        vm = mask_1d.view(B, 1, self.grid_h, self.grid_w)

        # Step 5: 3-stage ATB body with mask-aware attention
        x_2d = self.stage1(x_2d, vm, shift=False)

        # Dynamic mask update: relax validity over windows after stage 1
        w = self.stage1.attn.win
        vm_updated = (F.max_pool2d(vm, w, w) > 0).float()
        vm = vm_updated.repeat_interleave(w, -1).repeat_interleave(w, -2)

        x_2d = self.stage2(x_2d, vm, shift=True)
        x_2d = self.stage3(x_2d, vm, shift=False)

        # Step 6: Reverse 2D Bridge → (B, 128, 64)
        x_seq = x_2d.view(B, self.d_model, -1).permute(0, 2, 1)  # (B, 128, 64)

        # Step 7: Gather only the masked embeddings
        n_masks = masked_pos.size(1)
        gather_idx = masked_pos.unsqueeze(-1).expand(-1, -1, self.d_model)
        masked_emb = torch.gather(x_seq, 1, gather_idx)       # (B, n_masks, 64)

        # Step 8: Decode to patch space
        logits_masked = self.decoder(masked_emb)               # (B, n_masks, 16)

        # Step 9: MSE loss on masked patches
        loss = F.mse_loss(logits_masked, masked_tokens)

        return loss, logits_masked, masked_tokens
