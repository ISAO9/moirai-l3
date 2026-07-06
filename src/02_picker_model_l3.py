"""
02_picker_model_l3.py — MOIRAI L3 array-level P/S phase picker (2-D U-Net)
==========================================================================
WHAT THIS SCRIPT DOES
---------------------
Defines MoiraiPickerL3, the model that predicts per-pixel P / S / noise for a
WHOLE downhole vertical array at once. It is a direct transplant of the L2
RegressionUNet2D backbone (2-D U-Net, GroupNorm+SiLU ResBlocks, a light
bottleneck self-attention) with two task-driven changes:

  1. OUTPUT = 3 logit channels [P, S, noise] (probability picking), not the
     2 waveform channels of L2 separation.
  2. The image axes are (station, time): height = N_STATION (12, sparse),
     width = time. The depth axis carries the P/S MOVEOUT, which is the whole
     point of array-level picking -> we NEVER pool the 12-station axis. All
     down/up sampling is along TIME only, and the stem downsamples time by 2
     (L2 lesson: stem time-downsample of 4 destroyed P/S arrival resolution;
     2 was the fix). Conv kernels span the station axis (k=3) so moveout is
     mixed across depth through the encoder.

The model returns raw logits; activation (softmax for ce, sigmoid for
focal/bce) is provided via `.activate()` and applied by the loss (03) and the
evaluator (04). Backbone choices mirror L2 so its hard-won stability carries
over.

SELF-TEST (`python 02_picker_model_l3.py`): builds the model, prints the
parameter count, runs a forward+backward on a random batch, and asserts the
output is (B, N_PHASE, N_STATION, T) with the station/time axes preserved.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_module(filename: str):
    path = Path(__file__).with_name(filename)
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod  # register BEFORE exec so dataclasses can resolve types
    spec.loader.exec_module(mod)
    return mod


cfg = _load_module("00_config_l3.py")
DATA, MODEL, LOSS = cfg.DATA, cfg.MODEL, cfg.LOSS


# ----------------------------------------------------------------------------
# building blocks (station axis is NEVER strided; time axis is)
# ----------------------------------------------------------------------------
def _gn(ch: int) -> nn.GroupNorm:
    groups = 8 if (ch >= 8 and ch % 8 == 0) else 1
    return nn.GroupNorm(groups, ch)


class ResBlock2D(nn.Module):
    """Two 3x3 convs (span station + time) with GroupNorm+SiLU and a residual."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _gn(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (nn.Conv2d(in_ch, out_ch, kernel_size=1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class TimeDown(nn.Module):
    """Halve the time axis (stride 2), keep stations and channels."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=(1, 2), padding=1)

    def forward(self, x):
        return self.op(x)


class TimeUp(nn.Module):
    """Double the time axis exactly (k=(1,4), s=(1,2), p=(0,1)); keep stations."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, kernel_size=(1, 4),
                                     stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        return self.op(x)


class BottleneckAttention(nn.Module):
    """Single multi-head self-attention over flattened (station*time) tokens."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = _gn(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, s, t = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # (B, s*t, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).reshape(b, c, s, t)
        return x + h


# ----------------------------------------------------------------------------
# the picker
# ----------------------------------------------------------------------------
class MoiraiPickerL3(nn.Module):
    def __init__(self,
                 in_ch: int = MODEL.IN_CH,
                 out_ch: int = MODEL.OUT_CH,
                 base_ch: int = MODEL.BASE_CH,
                 ch_mult=MODEL.CH_MULT,
                 dropout: float = MODEL.DROPOUT,
                 use_attention: bool = MODEL.USE_ATTENTION,
                 stem_time_downsample: int = MODEL.STEM_TIME_DOWNSAMPLE):
        super().__init__()
        assert len(ch_mult) == MODEL.DEPTH, "len(CH_MULT) must equal DEPTH"
        chs = [base_ch * m for m in ch_mult]

        # stem: downsample time by `stem_time_downsample` (2), station untouched
        self.stem = nn.Conv2d(in_ch, base_ch, kernel_size=3,
                              stride=(1, stem_time_downsample), padding=1)

        # encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = base_ch
        for ch in chs:
            self.enc_blocks.append(ResBlock2D(prev, ch, dropout))
            self.downs.append(TimeDown(ch))
            prev = ch

        # bottleneck
        self.mid1 = ResBlock2D(prev, prev, dropout)
        self.mid_attn = BottleneckAttention(prev) if use_attention else nn.Identity()
        self.mid2 = ResBlock2D(prev, prev, dropout)

        # decoder (mirror)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for ch in reversed(chs):
            self.ups.append(TimeUp(prev))
            self.dec_blocks.append(ResBlock2D(prev + ch, ch, dropout))
            prev = ch

        # undo the stem time-downsample, then 1x1 head -> 3 logit channels
        self.stem_up = (TimeUp(prev) if stem_time_downsample == 2
                        else nn.Upsample(scale_factor=(1, stem_time_downsample),
                                         mode="nearest"))
        self.out_norm = _gn(prev)
        self.head = nn.Conv2d(prev, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, in_ch, N_STATION, T)
        t_in = x.shape[-1]
        h = self.stem(x)
        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            h = block(h)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid_attn(self.mid1(h)))
        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            # guard against any 1-sample rounding mismatch on the time axis
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1))
        h = self.stem_up(h)
        if h.shape[-1] != t_in:
            h = F.interpolate(h, size=(x.shape[-2], t_in), mode="nearest")
        logits = self.head(F.silu(self.out_norm(h)))
        return logits  # (B, out_ch, N_STATION, T)

    @staticmethod
    def activate(logits, loss_type: str = LOSS.TYPE):
        """Map logits to probabilities consistent with the training loss."""
        if loss_type == "ce":
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)


def build_model() -> MoiraiPickerL3:
    return MoiraiPickerL3()


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------
def selftest():
    print("=" * 74)
    print("MOIRAI L3 — 02 picker model self-test")
    print("=" * 74)
    torch.manual_seed(0)
    model = build_model()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_par/1e6:.2f} M")

    # small time length for a fast CPU check; architecture handles full window
    B, T = 2, 1024
    x = torch.randn(B, DATA.N_COMPONENT, DATA.N_STATION, T)
    y = torch.rand(B, DATA.N_PHASE, DATA.N_STATION, T)

    logits = model(x)
    print(f"  in : {tuple(x.shape)}")
    print(f"  out: {tuple(logits.shape)}")
    assert logits.shape == (B, DATA.N_PHASE, DATA.N_STATION, T), \
        f"output shape {tuple(logits.shape)} != (B,{DATA.N_PHASE},{DATA.N_STATION},{T})"
    assert logits.shape[-2] == DATA.N_STATION, "station axis not preserved"
    assert logits.shape[-1] == T, "time axis not preserved"

    # one backward to confirm gradient flow
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"  dummy BCE loss: {loss.item():.4f}   grad-norm-sum: {g:.3e}")
    assert g > 0, "no gradients flowed"

    proba = model.activate(logits.detach())
    print(f"  activate({LOSS.TYPE}) range: "
          f"[{proba.min():.3f}, {proba.max():.3f}]")
    print("  SHAPE + GRADIENT CHECK: PASS")


if __name__ == "__main__":
    selftest()
