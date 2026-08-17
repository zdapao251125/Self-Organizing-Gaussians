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
BC1_INTERP_WEIGHTS = [0.0, 1.0/3.0, 2.0/3.0, 1.0]  # BC1线性插值权重
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
    r = (rgb[..., 0] * 31.0).clamp(0, 31).round().to(torch.int16)
    g = (rgb[..., 1] * 63.0).clamp(0, 63).round().to(torch.int16)
    b = (rgb[..., 2] * 31.0).clamp(0, 31).round().to(torch.int16)

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
        # MPS backend does not support border padding in grid_sample.
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


def generate_crop_batch_correct(ref_base_res, max_lod, num_crops=1, crop_size=512, device="cuda"):
    # 每个 crop 内生成随机子像素偏移
    W = crop_size
    H = crop_size
    # 关键：每个像素位置加上一个随机偏移（0~1之间的小数）
    u = (torch.arange(W, device=device) + torch.rand(1, device=device)) / W
    v = (torch.arange(H, device=device) + torch.rand(1, device=device)) / H
    u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
    uv = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1)

    # LOD 采样保持不变（指数分布）
    if torch.rand(1).item() < 0.05:
        lod = torch.rand(1).item() * max_lod
    else:
        lod = (torch.rand(1).item() ** 4) * max_lod
    lod = torch.full((W * H,), lod, device=device)
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

        x_q = x_n + (ste_round(x_n * 3.0) / 3.0 - x_n).detach()  # 连续值 0, 1/3, 2/3, 1
        weights = x_q  # 直接作为插值权重

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
        e_qn = torch.round(e_n * ((1 << BC1_ENDPOINT_BITS) - 1)) / ((1 << BC1_ENDPOINT_BITS) - 1)
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

        self.decoder = MaterialDecoderMLP(
            in_dim=self.n_latent * 3 + 1,
            hidden=hidden_dim,
            out_dim=out_channels,
        )

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
        lod_norm = (lod / self.max_lod).unsqueeze(1)  # [B, 1]
        x = torch.cat([x, lod_norm], dim=1)  # [B, n_latent*3+1]
        out = self.decoder(x)
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


# Backward-compatible aliases for existing imports/scripts.
# UnconstrainedLatentPyramid = WarmupLatentPyramid
BCBlockMip = BC1SurrogateBlockLevel
BCBlockPyramid = BC1SurrogatePyramid
DecoderMLP = MaterialDecoderMLP
NeuralMaterialModel = NeuralMaterialCompressionModel


# -------------------------------
# Training config and loop
# -------------------------------

@dataclass
class TrainConfig:
    device: str = "cuda"
    batch_size: int = 4096 
    phase1_iters: int = 5_000
    phase2_iters: int = 200_000
    phase3_iters: int = 1_000

    lr_feat_phase1: float = 5e-2
    lr_mlp_phase1: float = 1e-3
    gamma_phase1: float = 0.9995

    lr_feat_phase2: float = 1e-2
    lr_mlp_phase2: float = 1e-3
    gamma_phase2: float = 0.9999

    lr_mlp_phase3: float = 5e-4

    log_every: int = 200
    interactive_progress: bool = False
    num_crops: int = 1
    crop_size: int = 256


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
        # Signed mode: latents in [-1, 1], convert to [0, 1] then uint8
        x = x.clamp(-1.0, 1.0)
        x = ((x + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
    else:
        # Unsigned mode: latents already in [0, 1], just convert to uint8
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

        # ---- 步骤1：强制 ep0 > ep1（避免透明模式） ----
        # if ep0 <= ep1:
        #     ep0, ep1 = ep1, ep0
        #     # 端点交换后，训练索引中的 0 和 3 代表的颜色互换
        #     swap_map = torch.tensor([1, 0, 3, 2], device=idx.device)
        #     idx = swap_map[idx]
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
        r = ((ep >> 11) & 0x1F) << 3
        g = ((ep >> 5) & 0x3F) << 2
        b = (ep & 0x1F) << 3
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
# Legacy single-subset BC6H helpers kept during the Mode 10 cleanup.
# The canonical target is spec-correct Mode 10 packing/export.
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
    state = model.decoder.state_dict()
    torch.save(state, meta_dir / "decoder_state.pt")
    flat = []
    for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"):
        flat.append(state[k].detach().to(torch.float16).contiguous().view(-1))
    fp16_blob = torch.cat(flat).cpu().numpy().tobytes()
    (out_dir / "decoder_fp16.bin").write_bytes(fp16_blob)

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
        "version": 4,
        "latent_count": model.n_latent,
        "latent_resolutions": model.latent_resolutions,
        "lod_biases": model.lod_biases,
        "bc_format": "BC1 (DXT1)",
        "bc_mode": 0,
        "endpoint_bits": 16,
        "index_bits": 2,
        "decoder": {
            "in_dim": int(model.decoder.fc1.in_features),
            "hidden_dim": int(model.decoder.fc1.out_features),
            "out_dim": int(model.decoder.fc3.out_features) if hasattr(model.decoder, 'fc3') else int(model.decoder.fc2.out_features),
            "weights_fp16_blob": "decoder_fp16.bin",
            "state_dict": "metadata/decoder_state.pt",
            "mlp_structure": "fc1→fc2→fc3" if hasattr(model.decoder, 'fc3') else "fc1→fc2",
        },
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

    # 检测MLP结构（自动适配单层/双层）
    state_dict = model.decoder.state_dict()
    has_fc3 = 'fc3.weight' in state_dict

    if has_fc3:
        out_channels = model.decoder.fc3.out_features
        hidden_dim = model.decoder.fc1.out_features
    else:
        out_channels = model.decoder.fc2.out_features
        hidden_dim = model.decoder.fc1.out_features

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
    # Override the material-specific Albedo transform: Gaussian attributes were
    # already normalised independently by compression/neural_texture.py.
    channel_mean.zero_()
    channel_std.fill_(1.0)
    ref_mips_norm = ref_mips
    channel_weight = 1.0 / raw_std
    channel_weight = torch.clamp(channel_weight, max=10.0)
    # Tile135D weight
    # Do not apply the original Tiles135D channel-specific weights to Gaussian data.

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

        uv, lod = generate_crop_batch_correct(ref_base_res=ref_base_res, max_lod=max_lod, num_crops=cfg.num_crops,
                                              crop_size=cfg.crop_size, device=device)

        target = sample_mips_trilinear(ref_mips_norm, uv, lod, bilinear_mode="bilinear")
        pred = _fwd_bc(uv, lod)

        # 3. 计算加权损失（替换原有的 loss_l1）
        loss = F.l1_loss(pred * channel_weight, target * channel_weight)
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

    print("[checkpoint] Phase 2 完成，保存中间状态...")
    _save_checkpoint(model, history, phase=2, iter=cfg.phase2_iters, export_dir=getattr(cfg, 'export_dir', None))

    # ---- Phase3: finetune
    if cfg.phase3_iters > 0:
        model.quantize_and_freeze_bc_features()
        opt3 = torch.optim.Adam(model.decoder_parameters(), lr=cfg.lr_mlp_phase3)
        phase3_iter = range(cfg.phase3_iters)
        pbar3 = tqdm(phase3_iter, desc="phase3", dynamic_ncols=True) if cfg.interactive_progress else None

        for it in (pbar3 or phase3_iter):
            uv, lod = generate_crop_batch_correct(ref_base_res=ref_base_res, max_lod=max_lod,num_crops=cfg.num_crops, crop_size=cfg.crop_size, device=device)

            # target = sample_mips_trilinear(ref_mips, uv, lod, bilinear_mode="bilinear")
            target = sample_mips_trilinear(ref_mips_norm, uv, lod, bilinear_mode="bilinear")
            pred = _fwd_bc(uv, lod)

            loss = F.l1_loss(pred * channel_weight, target * channel_weight)
            history.append({"phase": 3, "iter": it, "l1": float(loss.item())})

            opt3.zero_grad(set_to_none=True)
            loss.backward()
            opt3.step()

            if pbar3 and (it % (cfg.log_every // 2) == 0 or it == cfg.phase3_iters - 1):
                pbar3.set_postfix(l1=f"{loss.item():.4e}")

        if pbar3:
            pbar3.close()

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
    latent_resolutions = list(config.get("latent_resolutions", [1024, 1024, 512, 512]))
    latent_mips = list(config.get(
        "latent_mips", [int(math.log2(x)) + 1 for x in latent_resolutions]
    ))
    model = NeuralMaterialCompressionModel(
        latent_resolutions=latent_resolutions,
        latent_mips=latent_mips,
        out_channels=int(reference.shape[0]),
        hidden_dim=int(config.get("hidden_dim", 64)),
        ref_base_res=int(reference.shape[-1]),
        max_lod=float(len(ref_mips) - 1),
    ).to(device)
    cfg = TrainConfig(
        device=str(device),
        batch_size=int(config.get("batch_size", 4096)),
        phase2_iters=int(config.get("phase2_iters", 100000)),
        phase3_iters=int(config.get("phase3_iters", 1000)),
        lr_feat_phase2=float(config.get("lr_feat_phase2", 1e-2)),
        lr_mlp_phase2=float(config.get("lr_mlp_phase2", 1e-3)),
        gamma_phase2=float(config.get("gamma_phase2", 0.9999)),
        lr_mlp_phase3=float(config.get("lr_mlp_phase3", 5e-4)),
        log_every=int(config.get("log_every", 200)),
        interactive_progress=bool(config.get("interactive_progress", False)),
        num_crops=int(config.get("num_crops", 1)),
        crop_size=int(config.get("crop_size", 256)),
    )
    cfg.export_dir = Path(export_dir)
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
        print("开始导出BC压缩结果...")
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
