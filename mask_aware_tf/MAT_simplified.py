import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample_mask(m, h, w):
    # m: [N,1,H,W] or [N,2,H,W]; reduce to [N,1,H,W] then max-pool to (h,w)
    if m.size(1) > 1:
        m = (m.max(dim=1, keepdim=True).values > 0.5).float()
    kH, kW = m.size(-2)//h, m.size(-1)//w
    return F.max_pool2d(m, kernel_size=(kH, kW), stride=(kH, kW))

def make_attn_bias(valid_mask, tau=100.0):
    # valid_mask: [N,1,H,W] with {0,1}; -> [N,H*W,H*W] bias
    N, _, H, W = valid_mask.shape
    v = valid_mask.view(N, 1, H*W)                      # keys validities
    logits_bias = (1.0 - v) * (-tau)                    # 0 for valid, -tau for invalid
    # broadcast to query positions
    return logits_bias.expand(N, H*W, H*W)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.act  = nn.GELU()
    def forward(self, x): return self.act(self.conv(x)) 

class DSConvBlock(nn.Module):
    """Depthwise-separable version: DW 3x3 + PW 1x1 (same role, fewer FLOPs)."""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.dw  = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False)
        self.pw  = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(x)

class WindowMHA(nn.Module):
    def __init__(self, dim, heads=4, win=4):
        super().__init__()
        self.dim, self.heads, self.win = dim, heads, win
        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj   = nn.Linear(dim, dim, bias=False)

    def forward(self, x, valid_mask, shift=False, tau=100.0):
        # x: [N,C,H,W], valid_mask: [N,1,H,W]
        N, C, H, W = x.shape
        # optional shift
        if shift:
            s = self.win // 2
            x = torch.roll(x, shifts=(s, s), dims=(2, 3))
            valid_mask = torch.roll(valid_mask, shifts=(s, s), dims=(2, 3))
        # partition windows
        w = self.win
        assert H % w == 0 and W % w == 0
        xw = x.unfold(2,w,w).unfold(3,w,w)   # [N,C,H/w,W/w,w,w]
        xw = xw.contiguous().view(N, C, -1, w*w).permute(0,2,3,1)  # [N,nw,wsq,C]
        nw = xw.size(1)                      # number of windows
        # attention per-window (batched)
        qkv = self.to_qkv(xw).chunk(3, dim=-1)  # each [N,nw,wsq,C]
        def reshape(z):
            B, NW, S, D = z.shape
            Hh = self.heads
            return z.view(B*NW, S, Hh, D//Hh).permute(0,2,1,3)     # [B*NW,heads,S,dk]
        q, k, v = map(reshape, qkv)
        dk = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / (dk ** 0.5)             # [B*NW,heads,S,S]

        # build bias from mask per window
        vm = valid_mask.unfold(2,w,w).unfold(3,w,w)                # [N,1,H/w,W/w,w,w]
        vm = vm.contiguous().view(N, -1, 1, w*w)                   # [N,nw,1,S]
        bias = (1.0 - vm) * (-tau)                                 # [N,nw,1,S]
        bias = bias.repeat_interleave(self.heads, dim=2)           # [N,nw,heads,S]
        bias = bias.view(N*nw, self.heads, 1, w*w)                 # align with attn
        attn = attn + bias                                         # keys masking

        attn = attn.softmax(dim=-1)
        out = attn @ v                                             # [B*NW,heads,S,dk]
        out = out.permute(0,2,1,3).contiguous().view(N*nw, w*w, self.dim)
        out = self.proj(out)                                       # [N*nw,S,C]
        # fold windows back
        out = out.view(N, nw, w*w, C).permute(0,3,1,2)             # [N,C,nw,S]
        out = out.view(N, C, H//w, W//w, w, w).permute(0,1,2,4,3,5)
        out = out.contiguous().view(N, C, H, W)
        # reverse shift
        if shift:
            s = self.win // 2
            out = torch.roll(out, shifts=(-s, -s), dims=(2, 3))
        return out

class ATB(nn.Module):
    """Adjusted Transformer Block: attention -> concat -> FC -> MLP, plus local conv."""
    def __init__(self, dim, heads=4, win=4):
        super().__init__()
        self.attn = WindowMHA(dim, heads=heads, win=win)
        self.fc   = nn.Conv2d(dim*2, dim, 1)
        self.mlp  = nn.Sequential(nn.Conv2d(dim, dim*3, 1), nn.GELU(),
                                  nn.Conv2d(dim*3, dim, 1))
        self.local = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, valid_mask, shift=False):
        a = self.attn(x, valid_mask, shift=shift)        # attention (mask-aware)
        x = torch.cat([x, a], dim=1)                     # fusion by concat
        x = self.fc(x)
        x = x + self.mlp(x)                              # MLP
        x = x + self.local(x)                            # local conv
        return x

class MatTiny32x32(nn.Module):
    def __init__(self, c_in=2, c_out=2, width=64, heads=4, win=4, tau=100.0):
        super().__init__()
        self.tau = tau
        # head: 32 -> 16 -> 8
        self.h1 = ConvBlock(c_in, width, stride=1)
        self.h2 = ConvBlock(width, width, stride=2)
        self.h3 = ConvBlock(width, width, stride=2)

        # body: three stages, two ATBs each, shifted windows inside each stage
        self.b1 = nn.ModuleList([ATB(width, heads, win) for _ in range(1)])
        self.b2 = nn.ModuleList([ATB(width, heads, win) for _ in range(1)])
        self.b3 = nn.ModuleList([ATB(width, heads, win) for _ in range(1)])

        # tail: 8 -> 16 -> 32
        self.u1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                ConvBlock(width, width))
        self.u2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                ConvBlock(width, width))
        self.to_rgb = nn.Conv2d(width, c_out, 1)

    def forward(self, x_masked, mask):
        # mask to 1-channel
        if mask.size(1) > 1:
            mask = (mask.max(dim=1, keepdim=True).values > 0.5).float()
        # head
        x = self.h1(x_masked)      # 32x32
        x = self.h2(x)             # 16x16
        x = self.h3(x)             # 8x8

        vm = downsample_mask(mask, x.size(-2), x.size(-1))  # 8x8 validity

        # stage 1
        for i, blk in enumerate(self.b1):
            x = blk(x, vm, shift=(i % 2 == 1))
            # window-validity update: if any token valid in a window -> all valid
            vm = (F.max_pool2d(vm, 4, 4) > 0).float()
            vm = vm.repeat_interleave(4, -1).repeat_interleave(4, -2)  # back to 8x8

        # stage 2
        for i, blk in enumerate(self.b2):
            x = blk(x, vm, shift=(i % 2 == 1))

        # stage 3
        for i, blk in enumerate(self.b3):
            x = blk(x, vm, shift=(i % 2 == 1))

        # tail
        x = self.u1(x)             # 16x16
        x = self.u2(x)             # 32x32
        out = self.to_rgb(x)       # 2 channels
        return out
