# =============================================================================
# mat_pseudo_lwm.py — MAT Pseudo-Grid LWM (Method 2: 8×16 Pseudo-Grid)
#
# Aligns with the MAT paper architecture:
#   - Convolutional stem (no linear patch projection or positional embeddings)
#   - Mask-aware 2D spatial attention on the 8×16 pseudo-grid
#   - Continuous mask updating after every ATB stage
#
# Data flow:
#   (B,2,32,32) → ConvStem → (B,d_model,8,16) → spatial mask on grid
#   → 3-stage ATB (mask thawed after each stage) → decoder → MSE on masked
#
# Inference:
#   - CLS embedding:     GAP on pseudo-grid → (B, d_model)
#   - Channel embedding: flattened sequence  → (B, 128, d_model)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Normalization: Dynamic Tanh (Mask-Aware & Leak-Proof)
# =============================================================================

class DynamicTanh2d(nn.Module):
    """
    Dynamic Tanh normalizer for spatial tensors (B, C, H, W).
    Bounds activations to prevent exploding variance without causing spatial
    mask leakage that standard LayerNorm/InstanceNorm would create.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Learnable scale parameter per channel
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute normalized bounded activation
        return self.scale * torch.tanh(x / (self.scale.abs() + 1e-6))


# =============================================================================
# Adjusted Transformer Block (ATB) — from MAT, Dynamic Tanh Normalized
# =============================================================================

class ATB(nn.Module):
    """
    Adjusted Transformer Block: attention → concat → 1×1 FC → MLP + local conv.

    No LayerNorm is used, following the user specification and the original
    MAT design for inpainting where BN/LN can leak mask information.
    """

    def __init__(self, dim: int, heads: int = 4, win: int = 4):
        super().__init__()
        self.norm1 = DynamicTanh2d(dim)
        self.attn = WindowMHA(dim, heads=heads, win=win)
        self.fc = nn.Conv2d(dim * 2, dim, 1)                 # fuse concat
        self.norm2 = DynamicTanh2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 3, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim, 1),
        )
        self.norm3 = DynamicTanh2d(dim)
        self.local_conv = nn.Conv2d(dim, dim, 3, padding=1)   # local context

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
        a = self.attn(self.norm1(x), valid_mask, shift=shift)
        x = torch.cat([x, a], dim=1)                          # channel concat
        x = self.fc(x)                                        # fuse
        x = x + self.mlp(self.norm2(x))                       # feedforward
        x = x + self.local_conv(self.norm3(x))                 # local branch
        return x


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
    Hybrid MAT–LWM model using the 8×16 pseudo-grid approach (Method 2),
    aligned with the MAT paper architecture.

    Key design choices (MAT-aligned):
      - **Convolutional stem** instead of linear patch projection: two
        stacked 3×3 Conv2d layers downsample (B,2,32,32) → (B,d_model,8,16).
      - **No positional embeddings**: the conv stem and per-ATB local 3×3
        convolutions provide implicit relative positional encoding.
      - **Continuous mask updating**: the spatial validity mask is "thawed"
        after every ATB stage, not just once.

    Training mode (``gen_raw=False``):
        ConvStem → spatial mask on 8×16 grid → 3-stage ATB (mask thawed
        after each stage) → decoder → MSE loss on masked grid cells.

    Inference mode (``gen_raw=True``):
        ConvStem → 3-stage ATB (all-valid mask) → embeddings.

    Args:
        d_model:     Embedding / feature-map channels.  Default 64.
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
        d_model: int = 64,
        grid_h: int = 8,
        grid_w: int = 16,
        heads: int = 4,
        win: int = 4,
        mask_ratio: float = 0.15,
        gen_raw: bool = False,
        snr_db: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_patches = grid_h * grid_w                       # 128
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.mask_ratio = mask_ratio
        self.gen_raw = gen_raw
        self.snr_db = snr_db

        # ── Convolutional stem: (B,2,32,32) → (B,d_model,8,16) ──────────
        # Layer 1: stride (2,2) → 32×32 → 16×16, channels: 2 → d_model//2
        # Layer 2: stride (2,1) → 16×16 → 8×16,  channels: d_model//2 → d_model
        mid_ch = d_model // 2
        self.conv_stem = nn.Sequential(
            nn.Conv2d(2, mid_ch, kernel_size=3, stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, d_model, kernel_size=3, stride=(2, 1), padding=1),
            nn.GELU(),
        )

        # ── Learnable 2D positional embedding ─────────────────────────────
        self.pos_embed = nn.Parameter(
            torch.randn(1, d_model, grid_h, grid_w) * 0.02
        )

        # ── 3-stage ATB body (4×4 windows on 8×16 grid) ─────────────────
        self.stage1 = ATB(d_model, heads=heads, win=win)
        self.stage2 = ATB(d_model, heads=heads, win=win)
        self.stage3 = ATB(d_model, heads=heads, win=win)

        # ── Decoder head: project grid features back to 2-channel space ──
        # Upsample (B,d_model,8,16) → (B,2,32,32) for pixel-level loss
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, mid_ch, kernel_size=3,
                               stride=(2, 1), padding=1, output_padding=(1, 0)),
            nn.GELU(),
            nn.ConvTranspose2d(mid_ch, 2, kernel_size=3,
                               stride=(2, 2), padding=1, output_padding=1),
        )

    # ---- helpers --------------------------------------------------------
    def _thaw_mask(self, vm: torch.Tensor, win: int) -> torch.Tensor:
        """
        Relax validity mask: if *any* cell inside a ``win×win`` window is
        valid, mark the entire window valid.  Returns mask at the same
        spatial resolution as ``vm``.
        """
        vm_pooled = (F.max_pool2d(vm, win, win) > 0).float()   # (B,1,H/w,W/w)
        return vm_pooled.repeat_interleave(win, -1).repeat_interleave(win, -2)

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

    # ---- Spatial mask generation ----------------------------------------
    @staticmethod
    def _generate_spatial_mask(
        batch_size: int, h: int, w: int,
        mask_ratio: float, device: torch.device,
    ) -> torch.Tensor:
        """
        Generate a random binary spatial mask on the pseudo-grid.

        Uses LWM-compatible symmetric masking: the same row positions in
        the top half (real, rows 0-3) and bottom half (imag, rows 4-7)
        are masked together so real/imag patches stay paired.

        Args:
            batch_size: Number of samples.
            h, w:       Grid spatial dimensions (8, 16).
            mask_ratio: Fraction of grid cells to mask.
            device:     Target device.

        Returns:
            mask: (B, 1, h, w) float tensor.  1 = valid, 0 = masked.
        """
        half_h = h // 2                                        # 4
        n_cells_half = half_h * w                              # 64
        n_masked = max(1, int(mask_ratio * n_cells_half))

        mask = torch.ones(batch_size, 1, h, w, device=device)
        for i in range(batch_size):
            perm = torch.randperm(n_cells_half, device=device)[:n_masked]
            rows = perm // w
            cols = perm % w
            # Symmetric masking: same positions in real (top) and imag (bottom)
            mask[i, 0, rows, cols] = 0.0
            mask[i, 0, rows + half_h, cols] = 0.0

        return mask

    # ---- Forward --------------------------------------------------------
    def forward(self, channels: torch.Tensor):
        """
        Args:
            channels: (B, 2, 32, 32) real/imag wireless channel data.

        Returns (training, gen_raw=False):
            loss:           Scalar MSE loss on masked grid cells.
            pred_masked:    (N_masked,) predicted values at masked positions.
            target_masked:  (N_masked,) ground-truth values at masked positions.

        Returns (inference, gen_raw=True):
            cls_embedding:     (B, d_model)  global average pooled.
            channel_embedding: (B, 128, d_model) full sequence embeddings.
        """
        B = channels.size(0)

        # Optional noise augmentation
        if self.snr_db is not None:
            channels = self._add_complex_noise_ri(channels, self.snr_db)

        # ── Step 1: Convolutional stem ────────────────────────────────────
        x_2d = self.conv_stem(channels)                        # (B, d_model, 8, 16)
        x_2d = x_2d + self.pos_embed                           # add positional info

        # ── Inference mode ───────────────────────────────────────────────
        if self.gen_raw:
            vm = torch.ones(B, 1, self.grid_h, self.grid_w,
                            device=x_2d.device, dtype=x_2d.dtype)

            # 3-stage ATB body (full validity — no masking)
            x_2d = self.stage1(x_2d, vm, shift=False)
            x_2d = self.stage2(x_2d, vm, shift=True)
            x_2d = self.stage3(x_2d, vm, shift=False)

            # CLS embedding: GAP → (B, d_model)
            cls_emb = x_2d.mean(dim=(2, 3))

            # Channel embedding: (B, d_model, 8, 16) → (B, 128, d_model)
            channel_emb = x_2d.view(B, self.d_model, -1).permute(0, 2, 1)

            return cls_emb, channel_emb

        # ── Training mode ────────────────────────────────────────────────
        # Step 2: Generate spatial mask on the 8×16 grid
        mask = self._generate_spatial_mask(
            B, self.grid_h, self.grid_w,
            self.mask_ratio, channels.device,
        )                                                      # (B, 1, 8, 16)

        # Zero out masked positions in feature grid
        x_masked = x_2d * mask                                 # (B, d_model, 8, 16)
        vm = mask.clone()

        # Step 3: 3-stage ATB body with continuous mask updating ──────────
        # Stage 1
        x_masked = self.stage1(x_masked, vm, shift=False)
        w1 = self.stage1.attn.win
        vm = self._thaw_mask(vm, w1)                           # thaw after stage 1

        # Stage 2
        x_masked = self.stage2(x_masked, vm, shift=True)
        w2 = self.stage2.attn.win
        vm = self._thaw_mask(vm, w2)                           # thaw after stage 2

        # Stage 3 (uses the doubly-thawed mask)
        x_masked = self.stage3(x_masked, vm, shift=False)

        # Step 4: Decode back to channel space ────────────────────────────
        pred = self.decoder(x_masked)                          # (B, 2, 32, 32)

        # Step 5: Upsample mask to full resolution for loss computation ───
        # mask is (B, 1, 8, 16) → need (B, 1, 32, 32)
        mask_full = F.interpolate(
            mask, size=(32, 32), mode='nearest',
        )                                                      # (B, 1, 32, 32)

        # MSE only on masked pixels (where mask_full == 0)
        mask_2ch = mask_full.expand_as(channels)               # (B, 2, 32, 32)
        inverted = (mask_2ch == 0)                             # masked positions

        pred_masked = pred[inverted]
        target_masked = channels[inverted]

        loss = F.mse_loss(pred_masked, target_masked)

        return loss, pred_masked, target_masked
