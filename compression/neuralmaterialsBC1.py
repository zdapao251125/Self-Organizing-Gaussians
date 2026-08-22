#!/usr/bin/env python3
"""
Core training infrastructure for neural material compression.

Pipeline:
1) Warmup phase (unconstrained latent mip pyramids)
2) BC-constrained phase (differentiable BC6-style block parameterization + mip filtering)
3) Quantized finetune phase (freeze BC features, tune decoder on quantized path)

Notes:
- Training is being aligned with the paper's §4.2 / §5.2 BC6 workflow.
- The canonical runtime/export target is fixed-mode BC6H Mode 10 with official
  partitions, 6-bit endpoints, 3-bit indices, and FP16 decoder weights.
- `BC6_MODE10_TODO.md` tracks cleanup of legacy custom packing paths and stale
  Mode 11 / Mode 12 assumptions. PNG previews are kept for visual inspection.
- Export layout v4: runtime files in root (latent_XX.bc6.dds, decoder_fp16.bin,
  metadata.json); debug files in metadata/ (decoder_state.pt, latent_XX_mip_YY.png).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

BC1_ENDPOINT_BITS = 16
BC1_INDEX_BITS = 2
BC1_BLOCK_SIZE_BITS = 64  # 每个4x4块64位
# 标准四色BC1色阶，以端点1的重量表示：
# index 0 -> ep0, 1 -> ep1, 2 -> 2/3 ep0 + 1/3 ep1,
# index 3 -> 1/3 ep0 + 2/3 ep1.
BC1_INTERP_WEIGHTS = [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0]
BC1_RGB565_MASK = {
    'R': 0xF800, 'G': 0x07E0, 'B': 0x001F
}
BC1_RGB565_SHIFT = {
    'R': 11, 'G': 5, 'B': 0
}

# -------------------------------
# Utilities
# -------------------------------

EPS = 1e-8

GAUSSIAN_CHANNELS = (
    ("xyz", 0, 3),
    ("features_dc", 3, 6),
    ("features_rest", 6, 51),
    ("scaling", 51, 54),
    ("rotation", 54, 58),
    ("opacity", 58, 59),
)


def attribute_channels(num_channels: int):
    """Return the semantic channel layout used by the active compression target."""
    if num_channels == 45:
        return (("features_rest", 0, 45),)
    if num_channels == 59:
        return GAUSSIAN_CHANNELS
    raise ValueError(
        f"Unsupported Gaussian compression channel count: {num_channels}. "
        "Expected SH rest only (45) or all Gaussian attributes (59)."
    )


@torch.no_grad()
def debug_tensor(name: str, value: torch.Tensor) -> None:
    x = value.detach().float()
    finite = torch.isfinite(x)
    if not finite.all():
        print(f"[debug:{name}] shape={tuple(x.shape)} finite={finite.float().mean().item():.6f} "
              f"nan={torch.isnan(x).sum().item()} inf={torch.isinf(x).sum().item()}")
        x = x[finite]
    if x.numel() == 0:
        return
    print(f"[debug:{name}] shape={tuple(value.shape)} min={x.min().item():.6g} "
          f"max={x.max().item():.6g} mean={x.mean().item():.6g} "
          f"std={x.std(unbiased=False).item():.6g}")


@torch.no_grad()
def debug_compare(name: str, pred: torch.Tensor, target: torch.Tensor) -> None:
    p, t = pred.detach().float(), target.detach().float()
    if p.shape != t.shape:
        print(f"[debug:{name}] SHAPE MISMATCH pred={tuple(p.shape)} target={tuple(t.shape)}")
        return
    valid = torch.isfinite(p) & torch.isfinite(t)
    if not valid.all():
        print(f"[debug:{name}] non-finite pairs={(~valid).sum().item()}/{valid.numel()}")
        p, t = p[valid], t[valid]
    if p.numel() == 0:
        return
    error = p - t
    mse = error.square().mean()
    rmse = mse.sqrt()
    data_range = (t.max() - t.min()).clamp_min(EPS)
    psnr_range = 20.0 * torch.log10(data_range / rmse.clamp_min(EPS))
    psnr_unit = -10.0 * torch.log10(mse.clamp_min(EPS))
    print(f"[debug:{name}] mae={error.abs().mean().item():.6g} "
          f"rmse={rmse.item():.6g} max_abs={error.abs().max().item():.6g} "
          f"psnr_unit={psnr_unit.item():.3f}dB psnr_range={psnr_range.item():.3f}dB")


@torch.no_grad()
def debug_gaussian_batch(stage: str, pred: torch.Tensor, target: torch.Tensor) -> None:
    debug_tensor(f"{stage}/target", target)
    debug_tensor(f"{stage}/pred", pred)
    debug_compare(f"{stage}/all", pred, target)
    for attr, begin, end in attribute_channels(pred.shape[-1]):
        debug_compare(f"{stage}/{attr}", pred[:, begin:end], target[:, begin:end])


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def rgb_to_rgb565(rgb: torch.Tensor) -> torch.Tensor:
    """
    将归一化RGB张量 [*, 3] (0-1) 打包为RGB565 16bit整数
    Args:
        rgb: [*, 3] 浮点数张量 (0-1)
    Returns:
        [*] 整数张量 (0-65535)
    """
    # 量化到对应位数：R(5bit), G(6bit), B(5bit)
    # int16 有符号，当 RGB565 值的第 15 位被设置时，会变为负数，并导致后续整数打包损坏。因此将打包后的值保存为 int32。
    r = (rgb[..., 0] * 31.0).clamp(0, 31).round().to(torch.int32)
    g = (rgb[..., 1] * 63.0).clamp(0, 63).round().to(torch.int32)
    b = (rgb[..., 2] * 31.0).clamp(0, 31).round().to(torch.int32)

    # 打包为16bit整数：R<<11 | G<<5 | B
    rgb565 = (r << 11) | (g << 5) | b
    return rgb565


def rgb565_to_rgb(rgb565: torch.Tensor) -> torch.Tensor:
    """
    将RGB565 16bit整数张量解包为归一化RGB
    Args:
        rgb565: [*] 整数张量 (0-65535)
    Returns:
        [*, 3] 浮点数张量 (0-1)
    """
    # 解包各通道
    r = ((rgb565 >> 11) & 0x1F).to(torch.float32) / 31.0
    g = ((rgb565 >> 5) & 0x3F).to(torch.float32) / 63.0
    b = (rgb565 & 0x1F).to(torch.float32) / 31.0

    # 拼接为3通道
    rgb = torch.stack([r, g, b], dim=-1)
    return rgb

def quantize_ste(x: torch.Tensor, bits: int, min_v: float, max_v: float) -> torch.Tensor:
    if bits <= 0:
        return x
    levels = (1 << bits) - 1
    x = x.clamp(min_v, max_v)
    xn = (x - min_v) / (max_v - min_v + EPS)
    qn = ste_round(xn * levels) / levels
    return qn * (max_v - min_v) + min_v


def build_mip_chain(base_chw: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """Builds [mip0..mipN] from a base [C,H,W]."""
    mips = [base_chw]
    cur = base_chw
    for _ in range(levels - 1):
        h = max(1, cur.shape[1] // 2)
        w = max(1, cur.shape[2] // 2)
        cur = F.interpolate(
            cur.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mips.append(cur)
    return mips

#采样
def sample_texture_chw(tex_chw: torch.Tensor, uv: torch.Tensor, mode: str) -> torch.Tensor:
    """Samples [C,H,W] at uv [B,2] in [0,1], returns [B,C]."""
    sample_mode = mode
    if tex_chw.device.type == "mps":
        # MPS后端不支持grid_sample中的边框填充。
        uv = uv.clamp(1e-4, 1.0 - 1e-4)
        padding_mode = "zeros"
        if sample_mode == "bicubic":
            sample_mode = "bilinear"
    else:
        padding_mode = "border"
    grid = uv.mul(2.0).sub(1.0).view(1, -1, 1, 2)
    out = F.grid_sample(
        tex_chw.unsqueeze(0),
        grid,
        mode=sample_mode,
        align_corners=False,
        padding_mode=padding_mode,
    )
    return out.squeeze(0).squeeze(-1).transpose(0, 1)


def sample_discrete_texels(tex_chw: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Exact lookup for record-like Gaussian textures at texel-centre UVs."""
    height, width = int(tex_chw.shape[1]), int(tex_chw.shape[2])
    x = torch.floor(uv[:, 0] * width).long().clamp_(0, width - 1)
    y = torch.floor(uv[:, 1] * height).long().clamp_(0, height - 1)
    return tex_chw[:, y, x].T.contiguous()

#三线性采样
def sample_mips_trilinear(
    mips: Sequence[torch.Tensor], uv: torch.Tensor, lod: torch.Tensor, bilinear_mode: str
) -> torch.Tensor:
    """Trilinear over discrete mip levels; returns [B,C]."""
    max_lod = float(len(mips) - 1)
    lod = lod.clamp(0.0, max_lod - 1e-4)
    l0 = torch.floor(lod).long()
    l1 = torch.clamp(l0 + 1, max=len(mips) - 1)
    a = (lod - l0.float()).unsqueeze(1)

    all_samples = [sample_texture_chw(m, uv, mode=bilinear_mode) for m in mips]
    stack = torch.stack(all_samples, dim=0)  # [L,B,C]
    bidx = torch.arange(uv.shape[0], device=uv.device)
    v0 = stack[l0, bidx]
    v1 = stack[l1, bidx]
    return (1.0 - a) * v0 + a * v1


def random_uv_lod(batch: int, max_lod: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    uv = torch.rand(batch, 2, device=device)
    lod = torch.rand(batch, device=device) * max_lod
    return uv, lod


def generate_crop_batch_correct(
    ref_base_res,
    max_lod,
    num_crops=1,
    crop_size=512,
    device="cuda",
    uniform_sample_ratio=0.5,
    texel_center_sampling=False,
):
    """Generate ``num_crops`` independently jittered crops.

    A crop is placed on the reference pixel grid instead of always spanning the
    complete texture.  UV components are returned in (x, y) order.
    """
    side = min(int(crop_size), int(ref_base_res))
    all_uv, all_lod = [], []
    yy, xx = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device), indexing="ij"
    )
    for _ in range(int(num_crops)):
        max_origin = max(0, int(ref_base_res) - side)
        x0 = torch.randint(max_origin + 1, (), device=device) if max_origin else 0
        y0 = torch.randint(max_origin + 1, (), device=device) if max_origin else 0
        jitter = (
            torch.full((2,), 0.5, device=device)
            if texel_center_sampling else torch.rand(2, device=device)
        )
        u = (xx + x0 + jitter[0]) / float(ref_base_res)
        v = (yy + y0 + jitter[1]) / float(ref_base_res)
        all_uv.append(torch.stack((u.reshape(-1), v.reshape(-1)), dim=1))
        r = torch.rand((), device=device)
        sampled_lod = (r if torch.rand((), device=device) < 0.05 else r.pow(4)) * max_lod
        all_lod.append(sampled_lod.expand(side * side))
    uv = torch.cat(all_uv, dim=0)
    lod = torch.cat(all_lod, dim=0)

    # 统一裁剪起源会显著过采中心像素：边缘像素仅属于一个起源，而中心像素则属于约 crop_size 个起源。
    # 将可配置比例替换为全局均匀的像素采样，使边界/角落区域获得相等的训练概率。
    ratio = float(uniform_sample_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"uniform_sample_ratio must be in [0,1], got {ratio}")
    uniform_count = int(round(uv.shape[0] * ratio))
    if uniform_count:
        pixel_xy = torch.randint(
            int(ref_base_res), (uniform_count, 2), device=device
        )
        jitter = (
            torch.full((uniform_count, 2), 0.5, device=device)
            if texel_center_sampling
            else torch.rand(uniform_count, 2, device=device)
        )
        uv[:uniform_count] = (pixel_xy.float() + jitter) / float(ref_base_res)
    return uv, lod

class BC1SurrogateBlockLevel(nn.Module):
    """
    BC1-style block parameterization for one mip level:
    - per 4x4 block: 2 endpoints (RGB), 16 scalar indices
    """

    def __init__(self, h: int, w: int):
        super().__init__()
        assert h % 4 == 0 and w % 4 == 0, f"BC1块要求宽高为4的倍数，当前{h}x{w}"
        self.h = h
        self.w = w
        self.by = h // 4  # 块行数
        self.bx = w // 4  # 块列数
        self.nb = self.by * self.bx  # 总块数

        # BC1参数：每个块2个端点(RGB) + 16个2bit索引
        self.endpoints = nn.Parameter(torch.randn(self.nb, 2, 3) * 0.5)
        self.indices = nn.Parameter(torch.randn(self.nb, 16) * 0.5)

    @torch.no_grad()
    def init_from_unconstrained(self, mip_chw: torch.Tensor):
        """从无约束mip初始化BC1块参数（替代BC6的_mode10搜索）"""
        x = mip_chw.detach().clamp_min(0.0)
        device = x.device  # 获取输入张量的设备
        # [3,H,W] → [NB,16,3] 拆分4x4块
        blocks = (
            x.unfold(1, 4, 4)
            .unfold(2, 4, 4)
            .permute(1, 2, 3, 4, 0)
            .contiguous()
            .view(self.nb, 16, 3)
        )

        # BC1初始化逻辑：为每个块拟合2个端点+索引
        idx_levels = float((1 << BC1_INDEX_BITS) - 1)  # 2bit索引 → 0-3
        interp_weights = torch.tensor(BC1_INTERP_WEIGHTS, device=device)

        for i in range(self.nb):
            block = blocks[i]  # [16,3]
            # METHOD 3
            mean = block.mean(0, keepdim=True)
            X = block - mean
            cov = X.t() @ X
            eigvals, eigvecs = torch.linalg.eigh(cov)
            axis = eigvecs[:, -1]
            proj = X @ axis
            tmin = proj.min()
            tmax = proj.max()
            ep0 = mean[0] + tmin * axis
            ep1 = mean[0] + tmax * axis
            d = ep1 - ep0
            t = ((block - ep0) * d).sum(-1) / ((d * d).sum() + 1e-8)
            t = t.clamp(0, 1)

            # 匹配BC1的4个插值点
            idx = torch.argmin((t.unsqueeze(-1) - interp_weights).abs(), dim=-1)


            # 归一化并初始化参数（sigmoid反变换）
            ep_n = torch.stack([ep0, ep1], dim=0) / 1.0  # BC1端点范围[0,1]
            ep_n = ep_n.clamp(1e-4, 1.0 - 1e-4)
            self.endpoints[i].copy_(torch.log(ep_n / (1.0 - ep_n)))

            idx_n = idx.float() / idx_levels
            idx_n = idx_n.clamp(1e-4, 1.0 - 1e-4)
            self.indices[i].copy_(torch.log(idx_n / (1.0 - idx_n)))

    def _decode_soft_blocks(self) -> torch.Tensor:
        """修复：解码BC1块（正确的RGB565编解码）"""
        device = self.endpoints.device  # 获取设备
        # 1. 端点解码（RGB565反量化）
        e_n = torch.sigmoid(self.endpoints)  # [NB,2,3] → [0,1]

        # 分别量化 R,G,B 通道 (5,6,5 bit)
        r = e_n[..., 0]  # [NB,2]
        g = e_n[..., 1]
        b = e_n[..., 2]

        # 端点：前向量化，反向梯度直通
        r_q = r + (ste_round(r * 31.0) / 31.0 - r).detach()
        g_q = g + (ste_round(g * 63.0) / 63.0 - g).detach()
        b_q = b + (ste_round(b * 31.0) / 31.0 - b).detach()
        e_u = torch.stack([r_q, g_q, b_q], dim=-1)

        # 索引：前向量化到 {0,1,2,3}，反向梯度直通
        x_n = torch.sigmoid(self.indices)  # [NB,16] in [0,1]

        index_float = x_n * 3.0
        index_hard = ste_round(index_float)
        palette = x_n.new_tensor(BC1_INTERP_WEIGHTS)
        hard_weights = palette[index_hard.detach().long().clamp(0, 3)]
        # 在前向传播中使用标准的BC1配色方案，并通过连续索引参数进行直线梯度传递。
        weights = x_n + (hard_weights - x_n).detach()

        # 3. BC1线性插值计算
        ep0 = e_u[:, 0:1, :]  # [NB,1,3]
        ep1 = e_u[:, 1:2, :]  # [NB,1,3]
        y = ep0 * (1.0 - weights.unsqueeze(-1)) + ep1 * weights.unsqueeze(-1)  # [NB,16,3]

        return y  # [NB,16,3]


    def decode_mip(self) -> torch.Tensor:
        blocks = self._decode_soft_blocks()
        mip = (
            blocks.view(self.by, self.bx, 4, 4, 3)
            .permute(4, 0, 2, 1, 3)
            .contiguous()
            .view(3, self.h, self.w)
        )
        return mip

    @torch.no_grad()
    def export_quantized_block_params(self) -> dict:
        e_n = torch.sigmoid(self.endpoints)
        # e_n形状: [NB, 2, 3] → 需要打包为 [NB, 2] 的RGB565
        if e_n.dim() == 3 and e_n.shape[-1] == 3:
            # 将RGB浮点数转换为RGB565 16位整数
            e_q = rgb_to_rgb565(e_n)  # [NB, 2]
        else:
            e_q = torch.round(e_n * ((1 << BC1_ENDPOINT_BITS) - 1)).to(torch.int16)
        x_n = torch.sigmoid(self.indices)
        x_q = torch.round(x_n * ((1 << BC1_INDEX_BITS) - 1)).to(torch.uint8)
        # # BC1 只有在 endpoint0 > endpoint1 时才使用四色调色板。
        # 这里进行规范化，以确保导出的参数、打包字节和验证解码器具有完全相同的语义。
        swap = e_q[:, 0] <= e_q[:, 1]
        if swap.any():
            e_q = e_q.clone()
            x_q = x_q.clone()
            old0 = e_q[swap, 0].clone()
            e_q[swap, 0] = e_q[swap, 1]
            e_q[swap, 1] = old0
            swap_map = torch.tensor([1, 0, 3, 2], dtype=torch.uint8, device=x_q.device)
            x_q[swap] = swap_map[x_q[swap].long()]
        return {
            "h": self.h,
            "w": self.w,
            "endpoint_bits": BC1_ENDPOINT_BITS,
            "index_bits": BC1_INDEX_BITS,
            "endpoints_q": e_q.cpu(),
            "indices_q": x_q.cpu(),
        }

    @torch.no_grad()
    def quantize_inplace(self):
        # 端点量化（RGB565）
        e_n = torch.sigmoid(self.endpoints)
        e_qn = torch.stack((
            torch.round(e_n[..., 0] * 31.0) / 31.0,
            torch.round(e_n[..., 1] * 63.0) / 63.0,
            torch.round(e_n[..., 2] * 31.0) / 31.0,
        ), dim=-1)
        e_qn = e_qn.clamp(1e-4, 1.0 - 1e-4)
        self.endpoints.copy_(torch.log(e_qn / (1.0 - e_qn)))

        # 索引量化（2bit）
        x_n = torch.sigmoid(self.indices)
        x_qn = torch.round(x_n * ((1 << BC1_INDEX_BITS) - 1)) / ((1 << BC1_INDEX_BITS) - 1)
        x_qn = x_qn.clamp(1e-4, 1.0 - 1e-4)
        self.indices.copy_(torch.log(x_qn / (1.0 - x_qn)))


class BC1SurrogatePyramid(nn.Module):
    def __init__(self, base_res: int, num_mips: int):
        super().__init__()
        self.base_res = base_res
        self.num_mips = num_mips
        self.mips = nn.ModuleList()
        for i in range(num_mips):
            sz = max(4, base_res >> i)
            sz = (sz // 4) * 4  # BC1块对齐
            self.mips.append(BC1SurrogateBlockLevel(h=sz, w=sz))

    def decode_mips(self) -> List[torch.Tensor]:
        return [m.decode_mip() for m in self.mips]

    def sample(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        mips = self.decode_mips()
        return sample_mips_trilinear(mips, uv, lod, bilinear_mode="bilinear")

    @torch.no_grad()
    def export_quantized_params(self) -> List[dict]:
        return [m.export_quantized_block_params() for m in self.mips]

    @torch.no_grad()
    def quantize_inplace(self):
        for m in self.mips:
            m.quantize_inplace()


# -------------------------------
# Full model
# -------------------------------

class MaterialDecoderMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        # 1. 先进行常规的小随机初始化（此时还没训练）
        self._default_initialize()

        # 2. 执行“自定义数据驱动初始化”（这就是专家说的精髓）
        self._custom_data_driven_init()

    def _default_initialize(self):
        # 常规初始化，只需要极小的随机数就行
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0.0)

    def _custom_data_driven_init(self):
        """
        核心：基于随机输入，动态调整权重，确保激活后信号不衰减
        """
        device = next(self.parameters()).device

        # 1. 生成模拟输入（模拟真实训练时的 [Batch, 16] 维度）
        # 这里 batch 设为 1024，相当于跑一次前向测试
        with torch.no_grad():
            # 生成均值为0，方差为1的随机输入
            x = torch.randn(1024, self.fc1.in_features, device=device)

            # --- 处理第一层 fc1 ---
            out = self.fc1(x)
            # 如果是 ReLU 激活，我们确保激活前的输出标准差为 1
            # 这样才能保证 ReLU 后恰好一半为正
            std = out.std(unbiased=False)
            if std > 1e-6:
                # 关键步骤：将权重除以当前输出标准差，使得输出方差变为 1
                self.fc1.weight.data /= std
                self.fc1.bias.data /= std
                # 重新计算一下当前的激活值（用于下一层）
                out = self.fc1(x)

            # --- 处理第二层 fc2 ---
            out = F.silu(out)  # 这里假设你用 ReLU，如果用 SiLU 也类似
            out = self.fc2(out)
            std = out.std(unbiased=False)
            if std > 1e-6:
                self.fc2.weight.data /= std
                self.fc2.bias.data /= std

            # 最后一层 fc3 保持极小初始化即可（不做归一化，防止初期输出爆炸）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc3(x)


class NeuralMaterialCompressionModel(nn.Module):
    def __init__(
        self,
        latent_resolutions: Sequence[int],
        latent_mips: Sequence[int],
        out_channels: int,
        hidden_dim: int,
        ref_base_res: int,
        max_lod: float,
        scaling_decoder_hidden_dim: int = 64,
        features_rest_decoder_hidden_dim: int = 256,
        separate_gaussian_decoders: bool = True,
    ):
        super().__init__()
        assert len(latent_resolutions) == len(latent_mips)
        self.n_latent = len(latent_resolutions)
        self.latent_resolutions = list(latent_resolutions)
        self.ref_base_res = ref_base_res
        self.max_lod = max_lod  # 存储 max_lod 用于归一化

        self.bc_pyramids = nn.ModuleList(
            [
                BC1SurrogatePyramid(
                    base_res=r,
                    num_mips=m,
                )
                for r, m in zip(latent_resolutions, latent_mips)
            ]
        )

        decoder_in_dim = self.n_latent * 3 + 1
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.scaling_decoder_hidden_dim = int(scaling_decoder_hidden_dim)
        self.features_rest_decoder_hidden_dim = int(features_rest_decoder_hidden_dim)
        self.separate_gaussian_decoders = bool(
            separate_gaussian_decoders and out_channels == 59
        )
        if self.separate_gaussian_decoders:
            # 基础输出顺序: xyz(3), features_dc(3), rotation(4), opacity(1).
            self.decoder = MaterialDecoderMLP(decoder_in_dim, hidden_dim, 11)
            self.features_rest_decoder = MaterialDecoderMLP(
                decoder_in_dim, features_rest_decoder_hidden_dim, 45
            )
            self.scaling_decoder = MaterialDecoderMLP(
                decoder_in_dim, scaling_decoder_hidden_dim, 3
            )
        else:
            self.decoder = MaterialDecoderMLP(decoder_in_dim, hidden_dim, out_channels)

        # lod bias b_i = log2(max(h_i/h, w_i/w))
        self.lod_biases = [math.log2(max(r / ref_base_res, r / ref_base_res)) for r in latent_resolutions]
        self.freeze_bc_features = False
        self.register_buffer("channel_mean", torch.zeros(out_channels))
        self.register_buffer("channel_std", torch.ones(out_channels))


    @torch.no_grad()
    def initialize_bc_from_reference(self, ref_base: torch.Tensor):
        for pyr in self.bc_pyramids:
            pyr.init_from_reference(ref_base)

    def set_freeze_bc_features(self, enabled: bool):
        self.freeze_bc_features = enabled

    def bc_feature_parameters(self):
        for p in self.bc_pyramids.parameters():
            yield p

    def decoder_parameters(self):
        yield from self.decoder.parameters()
        if self.separate_gaussian_decoders:
            yield from self.features_rest_decoder.parameters()
            yield from self.scaling_decoder.parameters()

    def decoder_state_dicts(self):
        states = {"base": self.decoder.state_dict()}
        if self.separate_gaussian_decoders:
            states["features_rest"] = self.features_rest_decoder.state_dict()
            states["scaling"] = self.scaling_decoder.state_dict()
        return states

    def load_decoder_state_dicts(self, states):
        self.decoder.load_state_dict(states["base"])
        if self.separate_gaussian_decoders:
            self.features_rest_decoder.load_state_dict(states["features_rest"])
            self.scaling_decoder.load_state_dict(states["scaling"])

    def _decode_latent_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.separate_gaussian_decoders:
            return self.decoder(x)
        base = self.decoder(x)
        features_rest = self.features_rest_decoder(x)
        scaling = self.scaling_decoder(x)
        # 恢复标准的59通道高斯布局。
        return torch.cat(
            (base[:, 0:6], features_rest, scaling, base[:, 6:10], base[:, 10:11]),
            dim=1,
        )


    def _collect_latents_bc(self, uv: torch.Tensor, lod: torch.Tensor, use_uv_shift: bool = False) -> torch.Tensor:
        feats = []
        for i, (pyr, b) in enumerate(zip(self.bc_pyramids, self.lod_biases)):
            li = lod + b
            # 判断是否需要子像素偏移（索引 1 和 3）
            if use_uv_shift and (i == 1 or i == 3):
                # 半像素偏移量：0.5 / 基础分辨率
                offset = 0.5 / pyr.base_res  # pyr.base_res 是像素宽度
                uv_shifted = uv + offset  # 注意：xy 方向各加相同偏移
            else:
                uv_shifted = uv
            feats.append(pyr.sample(uv=uv_shifted, lod=li))
        return torch.cat(feats, dim=1)

    def forward_bc(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        x = self._collect_latents_bc(uv, lod, use_uv_shift=False)
        # 归一化 LOD 到 [0,1]
        lod_norm = (lod / max(float(self.max_lod), EPS)).unsqueeze(1)  # [B, 1]
        x = torch.cat([x, lod_norm], dim=1)  # [B, n_latent*3+1]
        out = self._decode_latent_input(x)
        # 反标准化（仅在推理时）
        if not self.training and hasattr(self, 'channel_mean'):
            out = out * self.channel_std + self.channel_mean
        return out

    @torch.no_grad()
    def quantize_and_freeze_bc_features(self):
        for pyr in self.bc_pyramids:
            pyr.quantize_inplace()
        for p in self.bc_feature_parameters():
            p.requires_grad_(False)
        self.set_freeze_bc_features(True)


# 与现有导入/脚本兼容的后向兼容别名。
# UnconstrainedLatentPyramid = WarmupLatentPyramid
BCBlockMip = BC1SurrogateBlockLevel
BCBlockPyramid = BC1SurrogatePyramid
DecoderMLP = MaterialDecoderMLP
NeuralMaterialModel = NeuralMaterialCompressionModel


# -------------------------------
# 训练配置和循环
# -------------------------------

@dataclass
class TrainConfig:
    device: str = "cuda"
    batch_size: int = 4096 
    phase1_iters: int = 5_000
    phase2_iters: int = 50_000
    phase3_iters: int = 1_000

    lr_feat_phase1: float = 5e-2
    lr_mlp_phase1: float = 1e-3
    gamma_phase1: float = 0.9995

    lr_feat_phase2: float = 1e-2
    lr_mlp_phase2: float = 1e-3
    gamma_phase2: float = 0.99994

    lr_mlp_phase3: float = 1e-4

    log_every: int = 200
    debug_every: int = 2000
    interactive_progress: bool = False
    num_crops: int = 1
    crop_size: int = 256
    uniform_sample_ratio: float = 1.0
    gaussian_texel_center_sampling: bool = True
    validation_res: int = 256
    full_validation_every: int = 5000
    full_validation_chunk_size: int = 65536
    full_validation_tile_size: int = 128
    full_validation_worst_tiles: int = 10
    full_validation_on_phase_end: bool = True
    full_validation_worst_tile_weight: float = 0.25
    full_validation_p99_weight: float = 0.10
    full_validation_opacity_weighted: bool = True
    features_rest_weight: float = 4.0
    features_rest_mse_weight: float = 5.0
    features_rest_variance_weight: float = 10.0
    features_rest_energy_weight: float = 2.0
    scaling_raw_weight: float = 2.0
    scaling_relative_weight: float = 0.25
    xyz_raw_weight: float = 0.25
    rotation_angle_weight: float = 2.0
    rotation_aligned_weight: float = 1.0
    rotation_norm_weight: float = 0.5
    phase3_validation_every: int = 100

    #新加
    lod0_only: bool = True


def load_reference_mips(path: Optional[Path], levels: int, out_channels: int, device: torch.device) -> List[torch.Tensor]:
    """
    Expects either:
    - .pt file with {"base": [C,H,W]} OR {"mips": List[[C,H,W]]}
    - if path is None, creates synthetic material channels for smoke testing.
    """
    if path is None:
        h = w = 1024
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        base = []
        for c in range(out_channels):
            f = torch.sin((c + 1) * math.pi * xx) * torch.cos((c + 1) * math.pi * yy)
            base.append(f)
        base = torch.stack(base, dim=0).to(torch.float32)
        return [m.to(device) for m in build_mip_chain(base, levels)]

    obj = torch.load(path, map_location="cpu")
    if "mips" in obj:
        mips = [m.float() for m in obj["mips"]]
    elif "base" in obj:
        mips = build_mip_chain(obj["base"].float(), levels)
    else:
        raise ValueError("Reference .pt must contain key 'mips' or 'base'.")

    if mips[0].shape[0] != out_channels:
        raise ValueError(f"Reference channels mismatch: got {mips[0].shape[0]}, expected {out_channels}.")
    return [m.to(device) for m in mips]


def save_chw_png_ldr(t: torch.Tensor, out_path: Path, signed_mode: bool = False):
    """Save CHW tensor to PNG LDR（BC1仅支持unsigned模式）"""
    if signed_mode:
        warnings.warn("BC1 does not support signed mode, forcing unsigned")
        signed_mode = False
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy are required to export PNG previews.")
    x = t.detach().to("cpu")
    if x.shape[0] < 3:
        x = x.repeat(3, 1, 1)[:3]
    x = x[:3]

    if signed_mode:
        # Signed mode: 将[-1, 1]范围内的latents转换为[0, 1]，然后转为uint8
        x = x.clamp(-1.0, 1.0)
        x = ((x + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
    else:
        # Unsigned mode: 已经在 [0, 1] 范围内的latents，只需转换为 uint8 即可
        x = x.clamp(0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8)

    img = x.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(img, mode="RGB").save(out_path)


def _pack_fields_to_fixed_block(fields: Sequence[Tuple[int, int]], total_bits: int = 128) -> bytes:
    """
    Packs integer fields into a fixed-size little-endian bitstream.
    Fields are appended in order, least-significant-bit first.
    """
    acc = 0
    offset = 0
    for value, nbits in fields:
        if nbits <= 0:
            continue
        mask = (1 << nbits) - 1
        acc |= (int(value) & mask) << offset
        offset += nbits
    if offset > total_bits:
        raise ValueError(f"bit overflow: used {offset} bits > {total_bits}")
    total_bytes = (total_bits + 7) // 8
    return int(acc).to_bytes(total_bytes, byteorder="little", signed=False)

def pack_quantized_blocks_to_64b(qp: dict) -> bytes:
    endpoints = qp["endpoints_q"].to(torch.int64)  # [NB, 2]
    indices = qp["indices_q"].to(torch.int64)      # [NB, 16]

    nb = endpoints.shape[0]
    out = bytearray()
    for i in range(nb):
        ep0 = int(endpoints[i, 0].item())
        ep1 = int(endpoints[i, 1].item())
        idx = indices[i].clone()  # 训练索引

        # 强制使用标准的四色BC1模式。当端点交换时，重新映射调色板索引，以确保所表示的颜色保持不变。
        if ep0 <= ep1:
            ep0, ep1 = ep1, ep0
            swap_map = torch.tensor([1, 0, 3, 2], device=idx.device)
            idx = swap_map[idx]
        #
        # # ---- 步骤2：训练索引 → 硬件索引 ----
        # # 映射规则：0→0, 1→2, 2→3, 3→1
        # new_idx = idx.clone()
        #
        # new_idx[idx == 0] = 0
        # new_idx[idx == 1] = 3
        # new_idx[idx == 2] = 1
        # new_idx[idx == 3] = 2
        #
        # idx = new_idx
        # ---- 打包 ----
        fields = []
        fields.append((ep0, 16))
        fields.append((ep1, 16))
        for x in idx.reshape(-1):
            fields.append((int(x.item()), 2))
        out.extend(_pack_fields_to_fixed_block(fields, total_bits=64))
    return bytes(out)


def unpack_quantized_blocks_from_64b_bc1(
        data: bytes, num_blocks: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """适配BC1：解包64bit块为「端点(RGB565) + 索引」"""
    rec_size = 8  # BC1块大小：8字节
    expected = num_blocks * rec_size
    if len(data) != expected:
        raise ValueError(f"Packed size mismatch: got {len(data)} expected {expected}")

    # BC1：2个RGB565端点 [NB,2], 16个2bit索引 [NB,16]
    endpoints = torch.empty((num_blocks, 2), dtype=torch.uint16)
    indices = torch.empty((num_blocks, 16), dtype=torch.uint8)

    for i in range(num_blocks):
        # 读取8字节BC1块（little-endian）
        block_data = data[i * rec_size: (i + 1) * rec_size]
        # 解析2个RGB565端点（前4字节=ep0，后4字节=ep1）
        ep0 = int.from_bytes(block_data[0:2], byteorder="little", signed=False)
        ep1 = int.from_bytes(block_data[2:4], byteorder="little", signed=False)
        endpoints[i] = torch.tensor([ep0, ep1], dtype=torch.uint16)

        # 解析16个2bit索引（后4字节，每个字节存4个索引）
        idx_bytes = block_data[4:8]
        idx_list = []
        for b in idx_bytes:
            # 每个字节拆分为4个2bit索引（LSB优先）
            idx_list.append((b >> 0) & 0x3)
            idx_list.append((b >> 2) & 0x3)
            idx_list.append((b >> 4) & 0x3)
            idx_list.append((b >> 6) & 0x3)
        indices[i] = torch.tensor(idx_list[:16], dtype=torch.uint8)
        # if i == 0:
        #     print(f"Block 0: ep0={ep0}, ep1={ep1}, idx={idx_list}")
        #     print(f"endpoints:{endpoints[i]},  idx={indices[i]}")

    return endpoints, indices

def unpack_and_decode_bc1_bytes(data: bytes, h: int, w: int) -> torch.Tensor:
    """解包BC1字节并解码为[3,H,W]张量"""
    nb = (h // 4) * (w // 4)
    endpoints, indices = unpack_quantized_blocks_from_64b_bc1(data, nb)
    # 转换为export_quantized_block_params的格式
    qp = {
        "h": h,
        "w": w,
        "endpoint_bits": BC1_ENDPOINT_BITS,
        "index_bits": BC1_INDEX_BITS,
        "endpoints_q": endpoints,
        "indices_q": indices,
    }
    return decode_bc1_params_to_mip(qp)

def decode_bc1_params_to_mip(qp: dict) -> torch.Tensor:
    """解码BC1量化参数为[3,H,W]张量"""
    if np is None:
        raise RuntimeError("numpy is required for BC1 decoding.")

    endpoints = qp["endpoints_q"].numpy()  # [NB,2] uint16 (RGB565)
    indices = qp["indices_q"].numpy()  # [NB,16] uint8
    h, w = qp["h"], qp["w"]
    bw, bh = w // 4, h // 4
    pixels = np.zeros((h, w, 3), dtype=np.uint8)

    # BC1解码核心：RGB565解包 + 2bit索引插值
    def rgb565_to_rgb(ep):
        # 解包RGB565：r(5bit), g(6bit), b(5bit) → 8bit
        # Match the normalized endpoint values used by the differentiable path.
        r = int(round(((ep >> 11) & 0x1F) * 255.0 / 31.0))
        g = int(round(((ep >> 5) & 0x3F) * 255.0 / 63.0))
        b = int(round((ep & 0x1F) * 255.0 / 31.0))
        return np.array([r, g, b], dtype=np.uint8)

    for by in range(bh):
        for bx in range(bw):
            bi = by * bw + bx
            ep0, ep1 = endpoints[bi]
            idx = indices[bi]
            # print(f"Block {bi}: ep0={ep0}, ep1={ep1}, idx={idx}")  # 打印输入
            # 解包端点
            rgb0 = rgb565_to_rgb(ep0).astype(np.int32)
            rgb1 = rgb565_to_rgb(ep1).astype(np.int32)
            # print(f"rgb0={rgb0}, rgb1={rgb1}")  # 打印解包后的颜色
            # 生成4x4像素（BC1插值规则）
            block_pixels = []
            for i in idx:
                if i == 0:
                    block_pixels.append(rgb0)
                elif i == 1:
                    block_pixels.append(rgb1)
                elif i == 2:
                    block_pixels.append((2 * rgb0 + rgb1) // 3)
                else:  # i==3
                    block_pixels.append((rgb0 + 2 * rgb1) // 3)
            block_pixels = np.array(block_pixels, dtype=np.uint8).reshape(4, 4, 3)

            # 写入对应位置
            y0, x0 = by * 4, bx * 4
            pixels[y0:y0 + 4, x0:x0 + 4] = block_pixels

    # 转为CHW格式（PyTorch常用）
    return torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32) / 255.0


# ---------------------------------------------------------------------------
# 在Mode 10清理期间保留下来的Legacy单子集BC6H辅助程序。  
# 标准目标是符合规范的Mode 10打包/导出。
# ---------------------------------------------------------------------------

import struct as _struct

# 新增BC1 DDS相关常量
_DXGI_FORMAT_BC1_UNORM = 71  # BC1 (DXT1) 无Alpha
_DDS_FOURCC_DXT1 = b"DXT1"
_DDS_MAGIC = b"DDS "
_DDSD_CAPS = 0x1
_DDSD_HEIGHT = 0x2
_DDSD_WIDTH = 0x4
_DDSD_PIXELFORMAT = 0x1000
_DDSD_MIPMAPCOUNT = 0x20000
_DDSD_LINEARSIZE = 0x80000
_DDSCAPS_COMPLEX = 0x8
_DDSCAPS_TEXTURE = 0x1000
_DDSCAPS_MIPMAP = 0x400000
_DDPF_FOURCC = 0x4

def _write_bc1_dds(mip_bytes_list: List[bytes],w0: int,h0: int,out_path: Path,):
    """写入BC1 (DXT1) 格式的DDS文件（替代BC6H的DDS写入）"""
    mip_count = len(mip_bytes_list)
    bw0, bh0 = max(1, (w0 + 3) // 4), max(1, (h0 + 3) // 4)
    linear_size = bw0 * bh0 * 8  # BC1每个块8字节

    # DDS头标志位
    flags = _DDSD_CAPS | _DDSD_HEIGHT | _DDSD_WIDTH | _DDSD_PIXELFORMAT | _DDSD_LINEARSIZE
    if mip_count > 1:
        flags |= _DDSD_MIPMAPCOUNT
    caps = _DDSCAPS_TEXTURE
    if mip_count > 1:
        caps |= _DDSCAPS_COMPLEX | _DDSCAPS_MIPMAP

    # 打包DDS头
    dds_header = _struct.pack(
        "<IIIIIII11I",
        124, flags, h0, w0, linear_size, 0, mip_count, *([0] * 11),
    )
    # 像素格式（DXT1 = BC1）
    dds_pixelformat = _struct.pack("<II4sIIIII", 32, _DDPF_FOURCC, _DDS_FOURCC_DXT1, 0, 0, 0, 0, 0)
    dds_caps = _struct.pack("<IIIII", caps, 0, 0, 0, 0)

    # 写入文件
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(_DDS_MAGIC)
        f.write(dds_header)
        f.write(dds_pixelformat)
        f.write(dds_caps)
        # 不需要DX10头（BC1是传统DDS格式）
        for mip_data in mip_bytes_list:
            f.write(mip_data)

@torch.no_grad()
def export_trained_artifacts(model: NeuralMaterialCompressionModel, out_dir: Path):
    """
    Export all runtime artifacts to out_dir (BC1/DXT1 VERSION).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # BC1 不支持 signed mode
    if hasattr(model, 'bc6_signed_mode') and model.bc6_signed_mode:
        raise NotImplementedError("BC1 only supports unsigned mode.")

    # --- Decoder weights ---
    decoder_modules = {"base": model.decoder}
    if model.separate_gaussian_decoders:
        decoder_modules["features_rest"] = model.features_rest_decoder
        decoder_modules["scaling"] = model.scaling_decoder
    decoder_meta = {}
    for branch, module in decoder_modules.items():
        state = module.state_dict()
        state_name = "decoder_state.pt" if branch == "base" else f"decoder_{branch}_state.pt"
        blob_name = "decoder_fp16.bin" if branch == "base" else f"decoder_{branch}_fp16.bin"
        torch.save(state, meta_dir / state_name)
        flat = []
        for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"):
            flat.append(state[k].detach().to(torch.float16).contiguous().view(-1))
        (out_dir / blob_name).write_bytes(torch.cat(flat).cpu().numpy().tobytes())
        decoder_meta[branch] = {
            "in_dim": int(module.fc1.in_features),
            "hidden_dim": int(module.fc1.out_features),
            "out_dim": int(module.fc3.out_features),
            "weights_fp16_blob": blob_name,
            "state_dict": f"metadata/{state_name}",
            "mlp_structure": "fc1→fc2→fc3",
        }

    # --- Latent DDS (BC1 from block params) + PNG previews ---
    latent_files = []
    model.set_freeze_bc_features(True)

    for i, pyr in enumerate(model.bc_pyramids):
        all_params = pyr.export_quantized_params()
        decoded_mips = pyr.decode_mips()

        mip_bytes_list = []
        for m, (params, tex) in enumerate(zip(all_params, decoded_mips)):
            stem = f"latent_{i:02d}_mip_{m:02d}"
            save_chw_png_ldr(tex, meta_dir / f"{stem}.png", signed_mode=False)

            # BC1 打包（64bit 块）
            packed_bytes = pack_quantized_blocks_to_64b(params)

            # 验证编码/解码一致性
            expected_pixels = decode_bc1_params_to_mip(params)
            decoded_pixels = unpack_and_decode_bc1_bytes(
                packed_bytes, h=params["h"], w=params["w"]
            )

            diff = expected_pixels.float() - decoded_pixels.float()
            max_abs = float(diff.abs().max().item())
            if max_abs > 1e-3:
                raise RuntimeError(f"BC1 pack/decode mismatch: max_abs={max_abs:.4e}")

            mip_bytes_list.append(packed_bytes)
            latent_files.append({
                "latent_index": i,
                "mip_index": m,
                "shape_chw": list(tex.shape),
                "png": f"metadata/{stem}.png",
            })

        # 写入 BC1 DDS
        W0, H0 = int(all_params[0]["w"]), int(all_params[0]["h"])
        dds_path = out_dir / f"latent_{i:02d}.bc1.dds"
        _write_bc1_dds(mip_bytes_list, W0, H0, dds_path)
        print(f"[export] BC1 latent {i:02d}: {W0}×{H0} → {dds_path.name}")

    meta = {
        "version": 5,
        "latent_count": model.n_latent,
        "latent_resolutions": model.latent_resolutions,
        "lod_biases": model.lod_biases,
        "bc_format": "BC1 (DXT1)",
        "bc_mode": 0,
        "endpoint_bits": 16,
        "index_bits": 2,
        "decoder_layout": "gaussian_split_v1" if model.separate_gaussian_decoders else "single",
        "decoders": decoder_meta,
        "decoder_output_mapping": {
            "base": ["xyz:0:3", "features_dc:3:6", "rotation:54:58", "opacity:58:59"],
            "features_rest": ["features_rest:6:51"],
            "scaling": ["scaling:51:54"],
        } if model.separate_gaussian_decoders else {"base": [f"all:0:{model.out_channels}"]},
        "latent_files": latent_files,
        "channel_mean": model.channel_mean.cpu().tolist(),
        "channel_std": model.channel_std.cpu().tolist(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] done → {out_dir}")


def _save_checkpoint(model, history, phase, iter, export_dir=None):
    """
    保存完整的训练checkpoint

    Args:
        model: 完整的模型
        history: 训练历史
        phase: 当前阶段 (1/2/3)
        iter: 当前迭代数
        export_dir: 导出目录（如果提供）
    """

    state_dict = model.decoder.state_dict()
    has_fc3 = 'fc3.weight' in state_dict
    out_channels = model.out_channels
    hidden_dim = model.hidden_dim

    # 构建checkpoint数据
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'phase': phase,
        'iter': iter,
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'latent_resolutions': model.latent_resolutions,
            'latent_mips': [p.num_mips for p in model.bc_pyramids],
            'out_channels': out_channels,
            'hidden_dim': hidden_dim,
            'scaling_decoder_hidden_dim': model.scaling_decoder_hidden_dim,
            'features_rest_decoder_hidden_dim': model.features_rest_decoder_hidden_dim,
            'separate_gaussian_decoders': model.separate_gaussian_decoders,
            'ref_base_res': model.ref_base_res,
            'n_latent': model.n_latent,
            'lod_biases': model.lod_biases,
            'mlp_layers': 3 if has_fc3 else 2,
            'max_lod': model.max_lod,
        }
    }

    # 确定保存路径
    if export_dir:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整checkpoint
        checkpoint_path = export_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ 完整模型checkpoint: {checkpoint_path}")

        # 同时保存一份带阶段标记的备份
        phase_checkpoint_path = export_dir / f"checkpoint_phase{phase}.pt"
        torch.save(checkpoint, phase_checkpoint_path)
        print(f"  ✓ 阶段备份: {phase_checkpoint_path}")

        # 保存训练历史为JSON（方便查看）
        history_path = export_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  ✓ 训练历史: {history_path}")

        # 保存配置为可读格式
        config_path = export_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(checkpoint['config'], f, indent=2)
        print(f"  ✓ 模型配置: {config_path}")
    else:
        # 如果没有导出目录，保存在当前目录
        checkpoint_path = Path(f"checkpoint_phase{phase}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint保存: {checkpoint_path}")

    # 返回checkpoint路径
    return checkpoint_path

def gaussian_attribute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    raw_minimum: torch.Tensor,
    raw_scale: torch.Tensor,
    channel_balance: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Hybrid normalized-domain and render-semantic Gaussian loss."""
    if pred.shape[-1] == 45 and target.shape[-1] == 45:
        pred_raw = pred * raw_scale + raw_minimum
        target_raw = target * raw_scale + raw_minimum
        balanced_l1 = F.l1_loss(
            pred * channel_balance,
            target * channel_balance,
        )
        features_rest_mse = F.mse_loss(pred, target)
        features_rest_variance = (
            pred.std(dim=0, unbiased=False)
            - target.std(dim=0, unbiased=False)
        ).abs().mean()
        features_rest_energy = (
            pred.square().mean(dim=0).sqrt()
            - target.square().mean(dim=0).sqrt()
        ).abs().mean()
        raw_l1 = F.smooth_l1_loss(pred_raw, target_raw)
        return (
            cfg.features_rest_weight * balanced_l1
            + cfg.features_rest_mse_weight * features_rest_mse
            + cfg.features_rest_variance_weight * features_rest_variance
            + cfg.features_rest_energy_weight * features_rest_energy
            + raw_l1
        )

    if pred.shape[-1] != 59 or target.shape[-1] != 59:
        raise ValueError(
            "gaussian_attribute_loss expects either 45 SH-rest channels or 59 full Gaussian channels "
            f"(pred={pred.shape[-1]}, target={target.shape[-1]}). "
            "Use a metadata-driven channel layout for other attribute sets."
        )
    weights = {
        "xyz": 20.0, "features_dc": 5.0,
        "features_rest": cfg.features_rest_weight,
        # 以下原始域术语具有最强的缩放/旋转监督。
        # 此处不要使用普通的组件L1进行旋转：q和-q是等价的，且与仅基于角度的目标冲突。
        # 旋转在下方通过符号对齐以及显式的范数项进行监督。
        "scaling": 2.0, "rotation": 0.0, "opacity": 10.0,
    }
    loss = sum(
        weights[name] * F.l1_loss(
            pred[:, begin:end] * channel_balance[:, begin:end],
            target[:, begin:end] * channel_balance[:, begin:end],
        )
        for name, begin, end in attribute_channels(pred.shape[-1])
    )

    pred_raw = pred * raw_scale + raw_minimum
    target_raw = target * raw_scale + raw_minimum

    # 带额外平方尾部惩罚的全局空间位置监督。
    xyz_delta = pred_raw[:, 0:3] - target_raw[:, 0:3]
    xyz_loss = F.smooth_l1_loss(pred_raw[:, 0:3], target_raw[:, 0:3])
    xyz_tail = xyz_delta.square().mean().sqrt()

    # _scaling 采用对数尺度。同时惩罚对数误差和乘法物理尺度比，同时限制指数以确保梯度稳定。
    scaling_delta = pred_raw[:, 51:54] - target_raw[:, 51:54]
    scaling_log_loss = F.smooth_l1_loss(
        pred_raw[:, 51:54], target_raw[:, 51:54]
    )
    scale_ratio = torch.exp(scaling_delta.clamp(-5.0, 5.0))
    scaling_relative_loss = F.smooth_l1_loss(scale_ratio, torch.ones_like(scale_ratio))

    pred_rest, target_rest = pred[:, 6:51], target[:, 6:51]
    features_rest_mse = F.mse_loss(pred_rest, target_rest)
    # Direct L1部分：对于相对较小的标准差差距，Smooth-L1 变为二次函数，但在数值上过于薄弱，无法防止方差崩溃。
    features_rest_variance = (
        pred_rest.std(dim=0, unbiased=False)
        - target_rest.std(dim=0, unbiased=False)
    ).abs().mean()
    features_rest_energy = (
        pred_rest.square().mean(dim=0).sqrt()
        - target_rest.square().mean(dim=0).sqrt()
    ).abs().mean()

    # q 和 -q 表示相同的旋转。对两边进行归一化，并优化它们的测地角距离，而不是仅使用分量上的 L1 距离。
    q_pred_raw = pred_raw[:, 54:58]
    q_target_raw = target_raw[:, 54:58]
    q_pred = F.normalize(q_pred_raw, dim=1, eps=EPS)
    q_target = F.normalize(q_target_raw, dim=1, eps=EPS)
    signed_dot = (q_pred * q_target).sum(dim=1, keepdim=True)
    # 在比较组件之前，先将预测的符号对齐到目标上。  
    # detach() 会将这个离散选择排除在梯度图之外。
    sign = torch.where(signed_dot.detach() < 0.0, -1.0, 1.0)
    q_pred_aligned = q_pred * sign
    rotation_aligned = F.smooth_l1_loss(q_pred_aligned, q_target)
    # 仅Angular损失对|q|不敏感，使得解码器输出能够接近零。保持原始解码的四元数长度接近单位长度。
    rotation_norm = F.smooth_l1_loss(
        q_pred_raw.norm(dim=1), torch.ones_like(q_pred_raw[:, 0])
    )
    dot = signed_dot.abs().squeeze(1).clamp(0.0, 1.0 - 1e-7)
    rotation_angle = (2.0 * torch.acos(dot)).mean()

    return (
        loss
        + cfg.xyz_raw_weight * (xyz_loss + 0.1 * xyz_tail)
        + cfg.scaling_raw_weight * scaling_log_loss
        + cfg.scaling_relative_weight * scaling_relative_loss
        + cfg.features_rest_mse_weight * features_rest_mse
        + cfg.features_rest_variance_weight * features_rest_variance
        + cfg.features_rest_energy_weight * features_rest_energy
        + cfg.rotation_angle_weight * rotation_angle
        + cfg.rotation_aligned_weight * rotation_aligned
        + cfg.rotation_norm_weight * rotation_norm
    )


@torch.no_grad()
def debug_error_quantiles(stage: str, pred: torch.Tensor, target: torch.Tensor) -> None:
    for attr, begin, end in attribute_channels(pred.shape[1]):
        # 每个样本/高斯误差的幅度相同，因此宽属性并不仅因通道更多而占据主导地位。
        err = (pred[:, begin:end] - target[:, begin:end]).square().mean(1).sqrt()
        qs = torch.quantile(err.float(), err.new_tensor([0.50, 0.95, 0.99, 0.999]))
        print(
            f"[quantile:{stage}/{attr}] p50={qs[0].item():.6g} "
            f"p95={qs[1].item():.6g} p99={qs[2].item():.6g} "
            f"p99.9={qs[3].item():.6g} max={err.max().item():.6g}"
        )


@torch.no_grad()
def debug_fixed_validation(model, ref_base, raw_minimum, raw_scale, cfg, stage: str) -> None:
    res = min(int(cfg.validation_res), int(ref_base.shape[-1]))
    height, width = int(ref_base.shape[1]), int(ref_base.shape[2])
    y_idx = torch.linspace(0, height - 1, res, device=ref_base.device).round().long()
    x_idx = torch.linspace(0, width - 1, res, device=ref_base.device).round().long()
    yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
    uv = torch.stack(
        ((xx.reshape(-1).float() + 0.5) / width,
         (yy.reshape(-1).float() + 0.5) / height), dim=1
    )
    lod = torch.zeros(uv.shape[0], device=uv.device)
    target = ref_base[:, yy.reshape(-1), xx.reshape(-1)].T.contiguous()
    pred = model.forward_bc(uv, lod)
    debug_gaussian_batch(f"{stage}/fixed_global_{res}x{res}", pred, target)
    debug_error_quantiles(f"{stage}/normalized", pred, target)
    pred_raw, target_raw = pred * raw_scale + raw_minimum, target * raw_scale + raw_minimum
    debug_error_quantiles(f"{stage}/raw", pred_raw, target_raw)
    if pred.shape[1] >= 58:
        q_pred = F.normalize(pred_raw[:, 54:58], dim=1, eps=EPS)
        q_target = F.normalize(target_raw[:, 54:58], dim=1, eps=EPS)
        angle = 2.0 * torch.acos((q_pred * q_target).sum(1).abs().clamp(0, 1 - 1e-7))
        aq = torch.quantile(angle * (180.0 / math.pi), angle.new_tensor([.5, .95, .99, .999]))
        print(f"[rotation-angle:{stage}] p50={aq[0].item():.3f}deg p95={aq[1].item():.3f}deg "
              f"p99={aq[2].item():.3f}deg p99.9={aq[3].item():.3f}deg "
              f"max={(angle.max() * 180.0 / math.pi).item():.3f}deg")


@torch.no_grad()
def fixed_validation_mse(model, ref_base: torch.Tensor, resolution: int) -> float:
    """Deterministic global metric used to select the best Phase-3 decoder."""
    res = min(int(resolution), int(ref_base.shape[-1]))
    height, width = int(ref_base.shape[1]), int(ref_base.shape[2])
    y_idx = torch.linspace(0, height - 1, res, device=ref_base.device).round().long()
    x_idx = torch.linspace(0, width - 1, res, device=ref_base.device).round().long()
    yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
    uv = torch.stack(
        ((xx.reshape(-1).float() + 0.5) / width,
         (yy.reshape(-1).float() + 0.5) / height), dim=1
    )
    lod = torch.zeros(uv.shape[0], device=uv.device)
    target = ref_base[:, yy.reshape(-1), xx.reshape(-1)].T.contiguous()
    pred = model.forward_bc(uv, lod)
    return float(F.mse_loss(pred, target).item())


@torch.no_grad()
def full_grid_validation(
    model, ref_base: torch.Tensor, raw_minimum: torch.Tensor,
    raw_scale: torch.Tensor, cfg: TrainConfig, stage: str,
) -> dict:
    """Decode every texel in bounded chunks and report tail/spatial metrics."""
    device = ref_base.device
    channels, height, width = ref_base.shape
    total = height * width
    chunk = max(1, int(cfg.full_validation_chunk_size))
    tile = max(1, int(cfg.full_validation_tile_size))
    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    tile_sse = torch.zeros(tiles_x * tiles_y, dtype=torch.float64)
    tile_count = torch.zeros(tiles_x * tiles_y, dtype=torch.int64)
    attr_errors = {
        name: torch.empty(total, dtype=torch.float32)
        for name, begin, end in attribute_channels(channels)
    }
    pixel_errors = torch.empty(total, dtype=torch.float32)
    rotation_angles = torch.empty(total, dtype=torch.float32) if channels >= 58 else None
    pred_sum = torch.zeros(channels, dtype=torch.float64)
    pred_sq_sum = torch.zeros(channels, dtype=torch.float64)
    target_sum = torch.zeros(channels, dtype=torch.float64)
    target_sq_sum = torch.zeros(channels, dtype=torch.float64)
    total_sse = 0.0
    weighted_sse = {"xyz": 0.0, "scaling": 0.0, "rotation": 0.0}
    weighted_den = 0.0
    was_training = model.training
    model.eval()

    ref_flat = ref_base.reshape(channels, -1)
    for begin_index in range(0, total, chunk):
        end_index = min(total, begin_index + chunk)
        flat = torch.arange(begin_index, end_index, device=device)
        y = torch.div(flat, width, rounding_mode="floor")
        x = flat - y * width
        uv = torch.stack(((x.float() + 0.5) / width, (y.float() + 0.5) / height), 1)
        lod = torch.zeros(end_index - begin_index, device=device)
        pred = model.forward_bc(uv, lod)
        target = ref_flat[:, begin_index:end_index].T
        delta = pred - target
        sq = delta.square()
        pred_cpu, target_cpu = pred.double().cpu(), target.double().cpu()
        pred_sum += pred_cpu.sum(0)
        pred_sq_sum += pred_cpu.square().sum(0)
        target_sum += target_cpu.sum(0)
        target_sq_sum += target_cpu.square().sum(0)

        pred_raw = pred * raw_scale + raw_minimum
        target_raw = target * raw_scale + raw_minimum
        for name, attr_begin, attr_end in attribute_channels(channels):
            err = (pred_raw[:, attr_begin:attr_end] - target_raw[:, attr_begin:attr_end])
            attr_errors[name][begin_index:end_index] = err.square().mean(1).sqrt().cpu()

        if channels >= 58:
            qp = F.normalize(pred_raw[:, 54:58], dim=1, eps=EPS)
            qt = F.normalize(target_raw[:, 54:58], dim=1, eps=EPS)
            angle = 2.0 * torch.acos((qp * qt).sum(1).abs().clamp(0, 1 - 1e-7))
            rotation_angles[begin_index:end_index] = (angle * 180.0 / math.pi).cpu()
            # q 和 -q 编码相同的旋转。
            # 在检查点评分中，用符号不变的单位-q 弦状误差替换普通编码的分量均方误差，同时保持四通道加权。
            q_mse = torch.minimum(
                (qp - qt).square().mean(1), (qp + qt).square().mean(1)
            )
            semantic_sq_sum = sq.sum(1) - sq[:, 54:58].sum(1) + 4.0 * q_mse
        else:
            semantic_sq_sum = sq.sum(1)
        per_pixel = semantic_sq_sum / channels
        pixel_errors[begin_index:end_index] = per_pixel.sqrt().cpu()
        total_sse += float(semantic_sq_sum.double().sum().item())
        tile_ids = ((y // tile) * tiles_x + (x // tile)).cpu()
        tile_sse += torch.bincount(
            tile_ids, weights=per_pixel.double().cpu(), minlength=tile_sse.numel()
        )
        tile_count += torch.bincount(tile_ids, minlength=tile_count.numel())
        if cfg.full_validation_opacity_weighted and channels >= 59:
            visibility = torch.sigmoid(target_raw[:, 58]).double()
            weighted_den += float(visibility.sum().item())
            weighted_sse["xyz"] += float(
                (visibility * (pred_raw[:, 0:3] - target_raw[:, 0:3]).square().mean(1).double()).sum().item()
            )
            weighted_sse["scaling"] += float(
                (visibility * (pred_raw[:, 51:54] - target_raw[:, 51:54]).square().mean(1).double()).sum().item()
            )
            weighted_sse["rotation"] += float(
                (visibility * angle.double().square()).sum().item()
            )

    global_mse = total_sse / max(total * channels, 1)
    global_rmse = math.sqrt(global_mse)
    global_psnr = -20.0 * math.log10(max(global_rmse, 1e-12))
    tile_mse = tile_sse / tile_count.clamp_min(1)
    worst_count = min(max(1, int(cfg.full_validation_worst_tiles)), tile_mse.numel())
    worst_values, worst_ids = torch.topk(tile_mse, worst_count)
    p99 = float(torch.quantile(pixel_errors, 0.99).item())
    score = (
        global_rmse
        + float(cfg.full_validation_worst_tile_weight) * math.sqrt(float(worst_values[0]))
        + float(cfg.full_validation_p99_weight) * p99
    )
    print(
        f"[full-validation:{stage}] grid={height}x{width} rmse={global_rmse:.7g} "
        f"psnr={global_psnr:.3f}dB p99={p99:.7g} score={score:.7g}"
    )
    for rank, (value, tile_id) in enumerate(zip(worst_values, worst_ids), 1):
        ty, tx = divmod(int(tile_id), tiles_x)
        print(
            f"[full-validation:{stage}/worst-tile-{rank}] x={tx * tile} y={ty * tile} "
            f"size={tile} rmse={math.sqrt(float(value)):.7g}"
        )
    qs_tensor = torch.tensor([.5, .95, .99, .999])
    for name, errors in attr_errors.items():
        qs = torch.quantile(errors, qs_tensor)
        print(
            f"[full-quantile:{stage}/{name}] p50={qs[0]:.6g} p95={qs[1]:.6g} "
            f"p99={qs[2]:.6g} p99.9={qs[3]:.6g} max={errors.max():.6g}"
        )
    if rotation_angles is not None:
        qs = torch.quantile(rotation_angles, qs_tensor)
        print(
            f"[full-rotation-angle:{stage}] p50={qs[0]:.3f}deg p95={qs[1]:.3f}deg "
            f"p99={qs[2]:.3f}deg p99.9={qs[3]:.3f}deg max={rotation_angles.max():.3f}deg"
        )
    pred_var = (pred_sq_sum / total - (pred_sum / total).square()).clamp_min(0)
    target_var = (target_sq_sum / total - (target_sum / total).square()).clamp_min(0)
    for name, attr_begin, attr_end in attribute_channels(channels):
        ratio = pred_var[attr_begin:attr_end].sqrt().mean() / target_var[attr_begin:attr_end].sqrt().mean().clamp_min(1e-12)
        print(f"[full-std-ratio:{stage}/{name}] pred_over_target={ratio.item():.6g}")
    if weighted_den > 0.0:
        print(
            f"[full-opacity-weighted:{stage}] xyz_rmse={math.sqrt(weighted_sse['xyz']/weighted_den):.7g} "
            f"scaling_rmse={math.sqrt(weighted_sse['scaling']/weighted_den):.7g} "
            f"rotation_rms={math.sqrt(weighted_sse['rotation']/weighted_den)*180/math.pi:.3f}deg"
        )
    if was_training:
        model.train()
    return {"score": score, "mse": global_mse, "rmse": global_rmse, "psnr": global_psnr}


@torch.no_grad()
def debug_training_stage(model, uv, lod, target, pred, stage: str) -> None:
    latents = model._collect_latents_bc(uv, lod, use_uv_shift=False)
    lod_norm = (lod / max(float(model.max_lod), EPS)).unsqueeze(1)
    decoder_input = torch.cat((latents, lod_norm), dim=1)
    decoder_raw = model._decode_latent_input(decoder_input)
    debug_tensor(f"{stage}/uv", uv)
    texel_coord = uv * float(model.ref_base_res)
    center_error = (texel_coord - torch.floor(texel_coord) - 0.5).abs()
    print(
        f"[debug:{stage}/texel_center_error] "
        f"mean={center_error.mean().item():.7g} max={center_error.max().item():.7g}"
    )
    debug_tensor(f"{stage}/lod", lod)
    debug_tensor(f"{stage}/bc1_latent_decode", latents)
    debug_tensor(f"{stage}/decoder_input", decoder_input)
    debug_compare(f"{stage}/forward_vs_decoder_raw", pred, decoder_raw)
    debug_gaussian_batch(f"{stage}/decoder_output", pred, target)
    if pred.shape[1] >= 51:
        pr, tr = pred[:, 6:51], target[:, 6:51]
        mse = F.mse_loss(pr, tr)
        std_l1 = (pr.std(0, unbiased=False) - tr.std(0, unbiased=False)).abs().mean()
        energy_l1 = (
            pr.square().mean(0).sqrt() - tr.square().mean(0).sqrt()
        ).abs().mean()
        ratio = pr.std(0, unbiased=False).mean() / tr.std(0, unbiased=False).mean().clamp_min(EPS)
        print(
            f"[features-rest-objective:{stage}] mse={mse.item():.7g} "
            f"std_l1={std_l1.item():.7g} energy_l1={energy_l1.item():.7g} "
            f"std_ratio={ratio.item():.6g}"
        )


def train(model: NeuralMaterialCompressionModel, ref_mips: List[torch.Tensor], cfg: TrainConfig) -> List[dict]:
    device = torch.device(cfg.device)
    max_lod = float(len(ref_mips) - 1)
    history: List[dict] = []

    _fwd_bc = model.forward_bc
    _fused = False

    # 获取参考纹理分辨率（假设正方形）
    ref_base_res = ref_mips[0].shape[2]

    # Method2
    ref_base = ref_mips[0]  # [C, H, W]
    C = ref_base.shape[0]
    raw_minimum = torch.as_tensor(
        getattr(cfg, "channel_minimum", [0.0] * C), device=device, dtype=ref_base.dtype
    ).view(1, -1)
    raw_scale = torch.as_tensor(
        getattr(cfg, "channel_scale", [1.0] * C), device=device, dtype=ref_base.dtype
    ).view(1, -1)
    if raw_minimum.shape[1] != C or raw_scale.shape[1] != C:
        raise ValueError(
            f"Normalization metadata mismatch: channels={C}, "
            f"minimum={raw_minimum.shape[1]}, scale={raw_scale.shape[1]}"
        )
    debug_tensor("reference/base_normalized", ref_base)
    for attr, begin, end in attribute_channels(C):
        debug_tensor(f"reference/{attr}", ref_base[begin:end])

    # 保留材料版本中对逆标准有用的平衡，但将其规范化到每个语义属性内部，防止其改变该属性的整体配置权重。
    # 这能够在不使低方差通道主导整个目标的情况下平衡复杂的SH通道。
    normalized_std = ref_base.std(dim=(1, 2), unbiased=False).clamp_min(1e-4)
    channel_balance = normalized_std.reciprocal()
    for attr, begin, end in attribute_channels(C):
        group = channel_balance[begin:end]
        channel_balance[begin:end] = (group / group.mean()).clamp(0.25, 4.0)
        print(
            f"[channel-balance:{attr}] min={channel_balance[begin:end].min().item():.4f} "
            f"max={channel_balance[begin:end].max().item():.4f} "
            f"mean={channel_balance[begin:end].mean().item():.4f}"
        )
    channel_balance = channel_balance.view(1, C)

    # --- 1.1 计算原始通道的标准差（用于后续加权损失）---
    raw_std = ref_base.std(dim=(1, 2), keepdim=False) + 1e-8  # [C]

    # --- 1.2 构建标准化的 mean 和 std（只对 Albedo 做 Z-score）---
    channel_mean = torch.zeros(C, 1, 1, device=device)
    channel_std = torch.ones(C, 1, 1, device=device)

    # 只对 Albedo（前三个通道，索引 0,1,2）计算真实的 mean/std
    for i in range(1,4):
        mean_i = ref_base[i].mean()
        std_i = ref_base[i].std() + 1e-8
        std_i = torch.clamp(std_i, min=0.1)  # 防止数值爆炸
        channel_mean[i, 0, 0] = mean_i
        channel_std[i, 0, 0] = std_i

    # print("channel_mean:", channel_mean)
    # print("channel_std:", channel_std)
    ref_mips_norm = [(mip - channel_mean) / channel_std for mip in ref_mips]
    # 覆盖材料特定的反照率变换：高斯属性已由 compression/neural_texture.py 独立归一化。
    channel_mean.zero_()
    channel_std.fill_(1.0)
    ref_mips_norm = ref_mips

    # 原始权重
    # channel_weight = 1.0 / raw_std
    # channel_weight = torch.clamp(channel_weight, max=10.0)

    channel_weight = torch.ones(C, device=device, dtype=ref_base.dtype)
    if C == 59:
        # Retained for compatibility with the full Gaussian experiment.
        channel_weight[0:3] = 20.0    # xyz
        channel_weight[3:6] = 5.0     # features_dc
        channel_weight[6:51] = 0.5    # features_rest
        channel_weight[51:54] = 10.0  # scaling
        channel_weight[54:58] = 5.0   # rotation
        channel_weight[58:59] = 10.0  # opacity
    else:
        # The SH-rest-only experiment has a local 45-channel layout.
        channel_weight.fill_(1.0)


    # Tile135D weight
    # 不要将原始的 Tiles135D 频道特定权重应用于高斯数据。

    # 将 mean 和 std 注册到模型（作为 buffer，不参与训练）
    model.channel_mean.copy_(channel_mean.squeeze())
    model.channel_std.copy_(channel_std.squeeze())

    # 计算每个 batch 的总像素数
    total_pixels = cfg.num_crops * cfg.crop_size * cfg.crop_size
    print(f"Using crop-based sampling: {cfg.num_crops} crops of {cfg.crop_size}x{cfg.crop_size} → {total_pixels} pixels per batch")

    # ---- Phase 2: BC1 constrained
    model.set_freeze_bc_features(False)
    opt2 = torch.optim.Adam(
        [
            {"params": list(model.bc_feature_parameters()), "lr": cfg.lr_feat_phase2},
            {"params": list(model.decoder_parameters()), "lr": cfg.lr_mlp_phase2},
        ],
        fused=_fused,
    )
    sch2 = torch.optim.lr_scheduler.ExponentialLR(opt2, gamma=cfg.gamma_phase2)

    phase2_iter = range(cfg.phase2_iters)
    pbar2 = tqdm(phase2_iter, desc="phase2", dynamic_ncols=True) if cfg.interactive_progress else None
    for it in (pbar2 or phase2_iter):

        uv, lod = generate_crop_batch_correct(
            ref_base_res=ref_base_res, max_lod=max_lod, num_crops=cfg.num_crops,
            crop_size=cfg.crop_size, device=device,
            uniform_sample_ratio=cfg.uniform_sample_ratio,
            texel_center_sampling=cfg.gaussian_texel_center_sampling,
        )

        #新加
        if cfg.lod0_only:
            lod.zero_()

        # 高斯网格是一组离散记录的表格，而非连续的物质图像。
        # 在LOD0时，亚像素抖动会混合相邻的高斯函数，并使模型趋向于人工平滑的目标。
        target = (
            sample_discrete_texels(ref_base, uv)
            if cfg.lod0_only and cfg.gaussian_texel_center_sampling
            else sample_mips_trilinear(ref_mips_norm, uv, lod, bilinear_mode="bilinear")
        )
        pred = _fwd_bc(uv, lod)

        # debug_every=0 disables periodic full-grid diagnostics; keep only
        # the first and final iteration in that mode.
        debug_now = (
            it == 0
            or it == cfg.phase2_iters - 1
            or (cfg.debug_every > 0 and it % cfg.debug_every == 0)
        )
        if debug_now:
            debug_training_stage(model, uv, lod, target, pred, f"phase2/iter_{it}")
            debug_fixed_validation(
                model, ref_base, raw_minimum, raw_scale, cfg, f"phase2/iter_{it}"
            )
        if cfg.full_validation_every > 0 and (it + 1) % cfg.full_validation_every == 0:
            full_grid_validation(
                model, ref_base, raw_minimum, raw_scale, cfg, f"phase2/iter_{it}"
            )

        # 3. 计算加权损失（替换原有的 loss_l1）
        loss = gaussian_attribute_loss(
            pred, target, raw_minimum, raw_scale, channel_balance, cfg
        )
        # 原始加权损失
        #loss = F.l1_loss(pred * channel_weight, target * channel_weight)

        # loss = F.l1_loss(pred, target)
        history.append({"phase": 2, "iter": it, "l1": float(loss.item())})

        opt2.zero_grad(set_to_none=True)
        loss.backward()
        #梯度检查
        # if it % 10000 == 0 and it > 0:
        #     for name, p in model.named_parameters():
        #         if 'bc' in name and p.grad is not None:
        #             print(f"[grad] {name}: {p.grad.norm().item():.4e}")
        opt2.step()
        sch2.step()

        if pbar2 and (it % cfg.log_every == 0 or it == cfg.phase2_iters - 1):
            pbar2.set_postfix(l1=f"{loss.item():.4e}")

    if pbar2:
        pbar2.close()

    if cfg.full_validation_on_phase_end:
        full_grid_validation(
            model, ref_base, raw_minimum, raw_scale, cfg, "phase2/end"
        )

    print("[checkpoint] Phase 2 完成，保存中间状态...")
    _save_checkpoint(model, history, phase=2, iter=cfg.phase2_iters, export_dir=getattr(cfg, 'export_dir', None))

    # ---- Phase3: finetune
    if cfg.phase3_iters > 0:
        model.quantize_and_freeze_bc_features()
        opt3 = torch.optim.Adam(model.decoder_parameters(), lr=cfg.lr_mlp_phase3)
        baseline_full = full_grid_validation(
            model, ref_base, raw_minimum, raw_scale, cfg, "phase3/baseline"
        )
        best_phase3_score = baseline_full["score"]
        best_decoder_state = {
            group: {key: value.detach().cpu().clone() for key, value in state.items()}
            for group, state in model.decoder_state_dicts().items()
        }
        best_phase3_iter = -1
        print(f"[phase3-best] baseline full-grid score={best_phase3_score:.8g}")
        phase3_iter = range(cfg.phase3_iters)
        pbar3 = tqdm(phase3_iter, desc="phase3", dynamic_ncols=True) if cfg.interactive_progress else None

        for it in (pbar3 or phase3_iter):
            uv, lod = generate_crop_batch_correct(
                ref_base_res=ref_base_res, max_lod=max_lod,
                num_crops=cfg.num_crops, crop_size=cfg.crop_size, device=device,
                uniform_sample_ratio=cfg.uniform_sample_ratio,
                texel_center_sampling=cfg.gaussian_texel_center_sampling,
            )

            #新加
            if cfg.lod0_only:
                lod.zero_()

            # target = sample_mips_trilinear(ref_mips, uv, lod, bilinear_mode="bilinear")
            target = (
                sample_discrete_texels(ref_base, uv)
                if cfg.lod0_only and cfg.gaussian_texel_center_sampling
                else sample_mips_trilinear(ref_mips_norm, uv, lod, bilinear_mode="bilinear")
            )
            pred = _fwd_bc(uv, lod)

            debug_now = (
                it == 0
                or it == cfg.phase3_iters - 1
                or (cfg.debug_every > 0 and it % cfg.debug_every == 0)
            )
            if debug_now:
                debug_training_stage(model, uv, lod, target, pred, f"phase3_quantized/iter_{it}")
                debug_fixed_validation(
                    model, ref_base, raw_minimum, raw_scale, cfg,
                    f"phase3_quantized/iter_{it}"
                )

            loss = gaussian_attribute_loss(
                pred, target, raw_minimum, raw_scale, channel_balance, cfg
            )
            # 原始加权损失
            #loss = F.l1_loss(pred * channel_weight, target * channel_weight)
            history.append({"phase": 3, "iter": it, "l1": float(loss.item())})

            opt3.zero_grad(set_to_none=True)
            loss.backward()
            opt3.step()

            validate_now = (
                (it + 1) % max(1, cfg.phase3_validation_every) == 0
                or it == cfg.phase3_iters - 1
            )
            if validate_now:
                validation = full_grid_validation(
                    model, ref_base, raw_minimum, raw_scale, cfg,
                    f"phase3/iter_{it}",
                )
                validation_score = validation["score"]
                print(f"[phase3-validation] iter={it} full-grid score={validation_score:.8g}")
                if validation_score < best_phase3_score:
                    best_phase3_score = validation_score
                    best_phase3_iter = it
                    best_decoder_state = {
                        group: {key: value.detach().cpu().clone() for key, value in state.items()}
                        for group, state in model.decoder_state_dicts().items()
                    }

            if pbar3 and (it % (cfg.log_every // 2) == 0 or it == cfg.phase3_iters - 1):
                pbar3.set_postfix(l1=f"{loss.item():.4e}")

        if pbar3:
            pbar3.close()
        model.load_decoder_state_dicts(best_decoder_state)
        print(
            f"[phase3-best] restored decoder from iter={best_phase3_iter} "
            f"full-grid score={best_phase3_score:.8g} "
            "(iter=-1 means the quantized Phase-2 baseline was best)"
        )

    # === 最终checkpoint保存 ===
    print("[checkpoint] 训练完成，保存最终模型状态...")
    _save_checkpoint(model, history, phase=3, iter=cfg.phase3_iters, export_dir=getattr(cfg, 'export_dir', None))

    return history

def train_from_tensor(reference: torch.Tensor, export_dir: Path, config=None):
    """Train directly from a normalised Gaussian attribute texture [C,H,W]."""
    config = dict(config or {})
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    reference = reference.detach().float().to(device)
    if reference.ndim != 3 or reference.shape[1] != reference.shape[2]:
        raise ValueError(f"Expected square [C,H,W] reference, got {tuple(reference.shape)}")

    ref_levels = int(config.get("ref_mips", int(math.log2(reference.shape[-1])) + 1))
    ref_mips = build_mip_chain(reference, ref_levels)
    # 高斯属性包含比原始9通道材料输入更多的信息。
    # 默认情况下，使用六个更高分辨率的BC1潜在变量（18个采样特征），而不是四个低分辨率的潜在变量（12个特征）。
    latent_resolutions = list(config.get(
        "latent_resolutions", [2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024]
    ))
    latent_mips = list(config.get(
        "latent_mips", [int(math.log2(x)) + 1 for x in latent_resolutions]
    ))
    model = NeuralMaterialCompressionModel(
        latent_resolutions=latent_resolutions,
        latent_mips=latent_mips,
        out_channels=int(reference.shape[0]),
        hidden_dim=int(config.get("hidden_dim", 192)),
        ref_base_res=int(reference.shape[-1]),
        max_lod=float(len(ref_mips) - 1),
        scaling_decoder_hidden_dim=int(config.get("scaling_decoder_hidden_dim", 64)),
        features_rest_decoder_hidden_dim=int(config.get("features_rest_decoder_hidden_dim", 256)),
        separate_gaussian_decoders=bool(config.get("separate_gaussian_decoders", True)),
    ).to(device)
    cfg = TrainConfig(
        device=str(device),
        batch_size=int(config.get("batch_size", 4096)),
        phase2_iters=int(config.get("phase2_iters", 50000)),
        phase3_iters=int(config.get("phase3_iters", 1000)),
        lr_feat_phase2=float(config.get("lr_feat_phase2", 1e-2)),
        lr_mlp_phase2=float(config.get("lr_mlp_phase2", 1e-3)),
        gamma_phase2=float(config.get("gamma_phase2", 0.99994)),
        lr_mlp_phase3=float(config.get("lr_mlp_phase3", 1e-4)),
        log_every=int(config.get("log_every", 200)),
        debug_every=int(config.get("debug_every", 2000)),
        interactive_progress=bool(config.get("interactive_progress", False)),
        num_crops=int(config.get("num_crops", 1)),
        crop_size=int(config.get("crop_size", 256)),
        uniform_sample_ratio=float(config.get("uniform_sample_ratio", 1.0)),
        gaussian_texel_center_sampling=bool(config.get("gaussian_texel_center_sampling", True)),
        validation_res=int(config.get("validation_res", 256)),
        full_validation_every=int(config.get("full_validation_every", 5000)),
        full_validation_chunk_size=int(config.get("full_validation_chunk_size", 65536)),
        full_validation_tile_size=int(config.get("full_validation_tile_size", 128)),
        full_validation_worst_tiles=int(config.get("full_validation_worst_tiles", 10)),
        full_validation_on_phase_end=bool(config.get("full_validation_on_phase_end", True)),
        full_validation_worst_tile_weight=float(config.get("full_validation_worst_tile_weight", 0.25)),
        full_validation_p99_weight=float(config.get("full_validation_p99_weight", 0.10)),
        full_validation_opacity_weighted=bool(config.get("full_validation_opacity_weighted", True)),
        features_rest_weight=float(config.get("features_rest_weight", 4.0)),
        features_rest_mse_weight=float(config.get("features_rest_mse_weight", 5.0)),
        features_rest_variance_weight=float(config.get("features_rest_variance_weight", 10.0)),
        features_rest_energy_weight=float(config.get("features_rest_energy_weight", 2.0)),
        scaling_raw_weight=float(config.get("scaling_raw_weight", 2.0)),
        scaling_relative_weight=float(config.get("scaling_relative_weight", 0.25)),
        xyz_raw_weight=float(config.get("xyz_raw_weight", 0.25)),
        rotation_angle_weight=float(config.get("rotation_angle_weight", 2.0)),
        rotation_aligned_weight=float(config.get("rotation_aligned_weight", 1.0)),
        rotation_norm_weight=float(config.get("rotation_norm_weight", 0.5)),
        phase3_validation_every=int(config.get("phase3_validation_every", 100)),

        #新加
        lod0_only=bool(config.get("lod0_only", True)),
    )
    cfg.export_dir = Path(export_dir)
    cfg.channel_minimum = list(config.get("channel_minimum", [0.0] * reference.shape[0]))
    cfg.channel_scale = list(config.get("channel_scale", [1.0] * reference.shape[0]))
    history = train(model, ref_mips, cfg)
    export_trained_artifacts(model, Path(export_dir))
    return model, history


def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-pt", type=Path, default=None)
    p.add_argument("--out-channels", type=int, default=13)  # BC1 = RGB 3通道
    p.add_argument("--ref-mips", type=int, default=13)
    p.add_argument("--latent-res", type=str, default="4096,4096,2048,2048")
    p.add_argument("--latent-mips", type=str, default="13,13,12,12")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=4096)
    # p.add_argument("--phase1-iters", type=int, default=15000)
    p.add_argument("--phase2-iters", type=int, default=850000)
    p.add_argument("--phase3-iters", type=int, default=1000)
    p.add_argument("--export-dir", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--interactive-progress", action="store_true")
    p.add_argument("--num-crops", type=int, default=1, help="Number of crops per batch (default: 8)")
    p.add_argument("--crop-size", type=int, default=256, help="Size of each crop in pixels (default: 256)")
    args = p.parse_args()

    device = torch.device(args.device)
    latent_res = parse_list_int(args.latent_res)
    latent_mips = parse_list_int(args.latent_mips)
    if len(latent_res) != len(latent_mips):
        raise ValueError("latent-res and latent-mips length mismatch")

    ref_mips = load_reference_mips(args.reference_pt, args.ref_mips, args.out_channels, device=device)
    ref_base_res = int(ref_mips[0].shape[2])

    # === BC1 不需要 partition_bank！===
    model = NeuralMaterialCompressionModel(
        latent_resolutions=latent_res,
        latent_mips=latent_mips,
        out_channels=args.out_channels,
        hidden_dim=args.hidden_dim,
        ref_base_res=ref_base_res,
        max_lod=len(ref_mips) - 1,
    ).to(device)

    cfg = TrainConfig(
        device=args.device,
        batch_size=args.batch_size,
        # phase1_iters=args.phase1_iters,
        phase2_iters=args.phase2_iters,
        phase3_iters=args.phase3_iters,
        log_every=args.log_every,
        interactive_progress=args.interactive_progress,
        num_crops=args.num_crops,
        crop_size=args.crop_size,
    )
    # 添加export_dir到cfg（用于checkpoint保存）
    if args.export_dir:
        cfg.export_dir = args.export_dir
    else:
        cfg.export_dir = None

    train(model, ref_mips, cfg)

    if args.export_dir is not None:
        print("\n" + "=" * 60)
        print("开始导出BC1压缩结果...")
        print("=" * 60)
        try:
            export_trained_artifacts(model=model, out_dir=args.export_dir)
            print("\n✓ 导出成功完成！")
        except Exception as e:
            print(f"\n✗ 导出时发生错误: {e}")
            print(f"但模型checkpoint已保存，可以手动重新导出")
            print(f"使用以下Python代码重新导出:")
            print(f"  from neuralmaterialsBC1 import *")
            print(f"  checkpoint = torch.load('{args.export_dir}/checkpoint.pt')")
            print(f"  model = NeuralMaterialCompressionModel(**checkpoint['config']).cuda()")
            print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
            print(f"  export_trained_artifacts(model, Path('{args.export_dir}'))")


if __name__ == "__main__":
    main()
