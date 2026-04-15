# =============================================================================
# mat_vit_lwm.py — MAT-ViT-LWM Hybrid Model (Method 1: 32×32 ViT Approach)
#
# Combines the Mask-Aware Transformer (MAT) architecture with LWM-compatible
# embeddings for wireless channel modelling. Operates directly on 32×32 spatial
# grids using masked self-supervised learning (no LWM 16-length patching).
#
# Architecture:
#   Input (B,2,32,32) → 1×1 Conv → (B,64,32,32) → Mask → 3-stage ATB body
#   → 1×1 Conv decoder → (B,2,32,32) → MSE on masked pixels
#
# For inference / downstream:
#   - CLS embedding:     GAP → (B, 64)
#   - Channel embedding: AdaptiveAvgPool2d(8,16) → (B, 128, 64)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helpers
# =============================================================================

def generate_spatial_mask(batch_size: int, h: int, w: int,
                          mask_ratio: float = 0.15,
                          device: torch.device = None) -> torch.Tensor:
    """
    Generate a random binary spatial mask.

    Args:
        batch_size: Number of samples in the batch.
        h, w:       Spatial dimensions.
        mask_ratio: Fraction of pixels to mask (set to 0).
        device:     Target device.

    Returns:
        mask: (B, 1, H, W) float tensor. 1 = valid, 0 = masked.
    """
    n_pixels = h * w
    n_masked = int(mask_ratio * n_pixels)

    # Create flat masks with random permutation per sample
    mask = torch.ones(batch_size, n_pixels, device=device)
    for i in range(batch_size):
        perm = torch.randperm(n_pixels, device=device)[:n_masked]
        mask[i, perm] = 0.0

    return mask.view(batch_size, 1, h, w)


# =============================================================================
# Window Multi-Head Attention (MCA) — from MAT
# =============================================================================

class WindowMHA(nn.Module):
    """
    Multi-Head Contextual Attention with shifted windows.

    Invalid (masked) tokens receive a large negative bias (``-tau``) so they
    are effectively ignored by the softmax, implementing the dynamic mask-aware
    strategy from the MAT paper.
    """

    def __init__(self, dim: int, heads: int = 4, win: int = 4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.win = win
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor,
                shift: bool = False, tau: float = 100.0) -> torch.Tensor:
        """
        Args:
            x:          (B, C, H, W) feature map.
            valid_mask: (B, 1, H, W) binary mask (1=valid, 0=masked).
            shift:      Whether to apply cyclic shift (shifted windows).
            tau:        Penalty magnitude for invalid tokens.

        Returns:
            out: (B, C, H, W) attention output.
        """
        N, C, H, W = x.shape
        w = self.win

        # Optional cyclic shift for shifted windows
        if shift:
            s = w // 2
            x = torch.roll(x, shifts=(s, s), dims=(2, 3))
            valid_mask = torch.roll(valid_mask, shifts=(s, s), dims=(2, 3))

        # Partition into non-overlapping windows
        assert H % w == 0 and W % w == 0, \
            f"Spatial dims ({H},{W}) must be divisible by window size {w}"

        # [N, C, H/w, w, W/w, w] → [N, C, nH, nW, w, w]
        xw = x.unfold(2, w, w).unfold(3, w, w)              # [N,C,nH,nW,w,w]
        xw = xw.contiguous().view(N, C, -1, w * w)           # [N,C,nw,wsq]
        xw = xw.permute(0, 2, 3, 1)                          # [N,nw,wsq,C]
        nw = xw.size(1)

        # QKV projection
        qkv = self.to_qkv(xw).chunk(3, dim=-1)               # 3 × [N,nw,wsq,C]

        def reshape(z):
            B, NW, S, D = z.shape
            Hh = self.heads
            return z.view(B * NW, S, Hh, D // Hh).permute(0, 2, 1, 3)  # [B*NW,heads,S,dk]

        q, k, v = map(reshape, qkv)
        dk = q.size(-1)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (dk ** 0.5)       # [B*NW,heads,S,S]

        # Build mask-aware bias: invalid keys get -tau penalty
        vm = valid_mask.unfold(2, w, w).unfold(3, w, w)       # [N,1,nH,nW,w,w]
        vm = vm.contiguous().view(N, -1, 1, w * w)            # [N,nw,1,S]
        bias = (1.0 - vm) * (-tau)                            # [N,nw,1,S]
        bias = bias.repeat_interleave(self.heads, dim=2)      # [N,nw,heads,S]
        bias = bias.view(N * nw, self.heads, 1, w * w)        # align with attn
        attn = attn + bias

        attn = attn.softmax(dim=-1)
        out = attn @ v                                         # [B*NW,heads,S,dk]

        # Merge heads and fold windows back
        out = out.permute(0, 2, 1, 3).contiguous().view(N * nw, w * w, self.dim)
        out = self.proj(out)                                   # [N*nw, wsq, C]

        # Reshape back to spatial
        out = out.view(N, nw, w * w, C).permute(0, 3, 1, 2)   # [N,C,nw,wsq]
        out = out.view(N, C, H // w, W // w, w, w)
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous()
        out = out.view(N, C, H, W)

        # Reverse cyclic shift
        if shift:
            s = w // 2
            out = torch.roll(out, shifts=(-s, -s), dims=(2, 3))

        return out


# =============================================================================
# Adjusted Transformer Block (ATB) — from MAT, no LayerNorm
# =============================================================================

class ATB(nn.Module):
    """
    Adjusted Transformer Block: attention → concat → 1×1 FC → MLP + local conv.

    No LayerNorm is used to prevent mask leakage.
    Instead, LayerScale is explicitly incorporated to prevent gradient/variance explosion
    across deep networks.
    """

    def __init__(self, dim: int, heads: int = 4, win: int = 4, init_values: float = 1e-4):
        super().__init__()
        self.attn = WindowMHA(dim, heads=heads, win=win)
        self.fc = nn.Conv2d(dim * 2, dim, 1)                 # fuse concat
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 3, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim, 1),
        )
        self.local_conv = nn.Conv2d(dim, dim, 3, padding=1)   # local context
        self.gamma_mlp = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))
        self.gamma_conv = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor,
                shift: bool = False) -> torch.Tensor:
        """
        Args:
            x:          (B, C, H, W) input features.
            valid_mask: (B, 1, H, W) validity mask.
            shift:      Whether to use shifted windows.

        Returns:
            x: (B, C, H, W) output features.
        """
        a = self.attn(x, valid_mask, shift=shift)
        x = torch.cat([x, a], dim=1)                          # channel concat
        x = self.fc(x)                                        # fuse
        x = x + self.gamma_mlp * self.mlp(x)                                   # feedforward
        x = x + self.gamma_conv * self.local_conv(x)                             # local branch
        return x


# =============================================================================
# MAT Stage
# =============================================================================

class MATStage(nn.Module):
    """
    MAT Stage containing 2 ATBs and a localized 3x3 Conv2d, wrapped in a LayerScaled
    residual connection. Reduced from 3→2 ATBs for better convergence at 100 epochs.
    """
    def __init__(self, dim: int, heads: int = 4, win: int = 4, init_values: float = 1e-4):
        super().__init__()
        self.block1 = ATB(dim, heads=heads, win=win, init_values=init_values)
        self.block2 = ATB(dim, heads=heads, win=win, init_values=init_values)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.gamma_stage = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor,
                shift: bool = False) -> torch.Tensor:
        """
        Alternates the shifted windows to ensure cross-window communication:
          block1: shift=False
          block2: shift=shift (argument)
        """
        identity = x
        x = self.block1(x, valid_mask, shift=False)
        x = self.block2(x, valid_mask, shift=shift)
        x = self.conv(x)
        return identity + self.gamma_stage * x


# =============================================================================
# MAT-ViT-LWM Hybrid Model
# =============================================================================

class MATViTLWM(nn.Module):
    """
    Hybrid MAT–ViT–LWM model for masked spatial pre-training on 32×32
    wireless channel matrices.

    Training mode (``gen_raw=False``):
        Accepts (B,2,32,32) channels, applies a random spatial mask,
        predicts the masked pixels, and returns the MSE loss.

    Inference mode (``gen_raw=True``):
        Returns LWM-compatible embeddings without masking.

    Args:
        width:       Embedding / feature-map width (D).  Default 64.
        heads:       Number of attention heads per ATB.  Default 4.
        win_sizes:   Window sizes per stage (s1, s2, s3).  Default (4, 8, 4).
        mask_ratio:  Fraction of pixels to mask during training.  Default 0.15.
        gen_raw:     If True, run in inference mode (no masking, return embeddings).
        snr_db:      Optional AWGN noise level (dB) applied before masking.
        tau:         Penalty for invalid tokens in attention.  Default 100.0.
    """

    def __init__(
        self,
        width: int = 64,
        heads: int = 4,
        win_sizes: tuple = (4, 8, 4),
        mask_ratio: float = 0.15,
        gen_raw: bool = False,
        snr_db: float | None = None,
        tau: float = 100.0,
    ):
        super().__init__()
        self.width = width
        self.mask_ratio = mask_ratio
        self.gen_raw = gen_raw
        self.snr_db = snr_db
        self.tau = tau

        # ── Initial projection: 2 channels → D ──────────────────────────
        self.proj_in = nn.Conv2d(2, width, kernel_size=1)

        # ── 3-stage ATB body (no LayerNorm, mask-aware) ─────────────────
        # Window sizes: stage1=4×4, stage2=8×8, stage3=4×4
        self.stage1 = MATStage(width, heads=heads, win=win_sizes[0])
        self.stage2 = MATStage(width, heads=heads, win=win_sizes[1])
        self.stage3 = MATStage(width, heads=heads, win=win_sizes[2])

        # ── Decoder head: D → 2 channels ────────────────────────────────
        self.proj_out = nn.Conv2d(width, 2, kernel_size=1)

        # ── Embedding extraction (inference) ─────────────────────────────
        self.pool_channel = nn.AdaptiveAvgPool2d((8, 16))

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
    def forward(self, channels: torch.Tensor,
                mask: torch.Tensor | None = None):
        """
        Args:
            channels: (B, 2, 32, 32) real/imag wireless channel data.
            mask:     Optional (B, 1, 32, 32) binary mask. If None and
                      ``gen_raw=False``, a random mask is generated.

        Returns (training, gen_raw=False):
            loss:           Scalar MSE loss on masked pixels.
            pred_masked:    (N_masked,) predicted values at masked positions.
            target_masked:  (N_masked,) ground-truth values at masked positions.

        Returns (inference, gen_raw=True):
            cls_embedding:     (B, 64)    global average pooled.
            channel_embedding: (B, 128, 64) LWM-compatible sequence.
        """
        B, _, H, W = channels.shape

        # Optional noise augmentation
        if self.snr_db is not None:
            channels = self._add_complex_noise_ri(channels, self.snr_db)

        # Projection to embedding space
        x = self.proj_in(channels)                             # (B, D, 32, 32)

        # ── Inference mode ───────────────────────────────────────────────
        if self.gen_raw:
            # Full-validity mask (all ones)
            vm = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)

            # 3-stage ATB body (no shifting needed in inference)
            x = self.stage1(x, vm, shift=False)
            x = self.stage2(x, vm, shift=False)
            x = self.stage3(x, vm, shift=False)

            # CLS embedding: Global Average Pooling → (B, D)
            cls_emb = x.mean(dim=(2, 3))                      # (B, 64)

            # Channel embedding: pool to (8,16) → flatten → (B, 128, D)
            pooled = self.pool_channel(x)                      # (B, 64, 8, 16)
            # Reshape: (B, D, 8, 16) → (B, D, 128) → (B, 128, D)
            channel_emb = pooled.view(B, self.width, -1)       # (B, 64, 128)
            channel_emb = channel_emb.permute(0, 2, 1)         # (B, 128, 64)

            return cls_emb, channel_emb

        # ── Training mode ────────────────────────────────────────────────
        # Generate or use provided mask
        if mask is None:
            mask = generate_spatial_mask(
                B, H, W, mask_ratio=self.mask_ratio, device=channels.device
            )                                                  # (B, 1, H, W)

        # Apply mask to projected features (zero out masked positions)
        x_masked = x * mask                                    # (B, D, 32, 32)

        # 3-stage ATB body with dynamic mask updating
        # Stage 1: with mask validity update (window propagation)
        vm = mask.clone()
        x_masked = self.stage1(x_masked, vm, shift=False)

        # Dynamic mask update: if any token in a window is valid → all valid
        # Use stage 2's window size for the validity propagation
        w2 = self.stage2.block1.attn.win
        vm_updated = (F.max_pool2d(vm, w2, w2) > 0).float()
        vm = vm_updated.repeat_interleave(w2, -1).repeat_interleave(w2, -2)

        # Stage 2 & 3: with updated (relaxed) mask
        x_masked = self.stage2(x_masked, vm, shift=True)
        x_masked = self.stage3(x_masked, vm, shift=False)

        # Decode back to channel space
        pred = self.proj_out(x_masked)                         # (B, 2, 32, 32)

        # ── Loss: MSE only on masked pixels ──────────────────────────────
        # Expand mask to 2-channel: (B, 1, H, W) → (B, 2, H, W)
        mask_2ch = mask.expand_as(channels)                    # (B, 2, 32, 32)
        inverted = (mask_2ch == 0)                             # masked positions

        pred_masked = pred[inverted]
        target_masked = channels[inverted]

        loss = F.mse_loss(pred_masked, target_masked)

        return loss, pred_masked, target_masked
