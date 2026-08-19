from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

# 取得当前文件上两级目录作为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入neuralmaterialsBC1，如果该文件移位，记得修改前面的路径
from compression.neuralmaterialsBC1 import (  # noqa: E402
    sample_mips_trilinear,
    train_from_tensor,
)

# 需要压缩的 Gaussian 属性
DEFAULT_ATTRIBUTES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
)


def _attribute_names(config: Dict[str, Any]) -> Iterable[str]:

    # 从配置中取得需要压缩的 Gaussian 属性名称
    attributes = config.get("attributes")

    # 如果配置中没有提供 attributes，则使用 DEFAULT_ATTRIBUTES 中定义的默认属性
    if not attributes:
        return DEFAULT_ATTRIBUTES
    return [item["name"] if isinstance(item, dict) else item for item in attributes]

# 数据归一化
def _normalise_chw(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    minimum = image.amin(dim=(1, 2), keepdim=True)
    maximum = image.amax(dim=(1, 2), keepdim=True)
    scale = (maximum - minimum).clamp_min(1e-8)
    return (image - minimum) / scale, minimum, scale

# 计算目录内所有普通文件的总大小
def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _runtime_directory_size(path: Path) -> int:
    """Count only files needed to decode the compressed representation."""
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = set(metadata.get("runtime_files", []))
    files.add("gaussian_layout.json")
    return sum((path / name).stat().st_size for name in files if (path / name).is_file())

# 压缩
def compress_gaussians(gaussians, out_dir: str | os.PathLike, config=None) -> int:
    config = dict(config or {})
    out_dir = Path(out_dir)

    # 创建输出目录
    out_dir.mkdir(parents=True, exist_ok=True)

    # 记录当前已占用的通道数量
    channel_offset = 0

    # 保存每个属性归一化后的 CHW 张量
    normalised_images = []

    # 保存各属性的通道位置、原始形状和归一化参数
    layout = {}

    # 遍历需要压缩的 Gaussian 属性
    for name in _attribute_names(config):
        tensor = getattr(gaussians, name)
        original_shape = list(tensor.shape[1:])
        grid = gaussians.attr_as_grid_img(name)
        height, width = int(grid.shape[0]), int(grid.shape[1])
        chw = grid.reshape(height, width, -1).permute(2, 0, 1).detach().float().cpu()
        normalised, minimum, scale = _normalise_chw(chw)

        # 当前属性展开后的通道数量
        channels = int(chw.shape[0])

        # 记录当前属性在最终拼接纹理中的布局
        layout[name] = {
            "start": channel_offset,
            "end": channel_offset + channels,
            "tensor_tail_shape": original_shape,
            "minimum": minimum.flatten().tolist(),
            "scale": scale.flatten().tolist(),
        }
        channel_offset += channels
        normalised_images.append(normalised)

    if not normalised_images:
        raise ValueError("Neural texture compression needs at least one attribute.")

    # 沿通道维拼接所有属性
    reference = torch.cat(normalised_images, dim=0)

    # 创建解压时需要使用的元数据
    metadata = {
        "format_version": 1,
        "height": int(reference.shape[1]),
        "width": int(reference.shape[2]),
        "channels": int(reference.shape[0]),
        "attributes": layout,
    }
    (out_dir / "gaussian_layout.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # 使用拼接后的多通道纹理训练神经压缩模型
    train_from_tensor(reference, out_dir, config)

    # Checkpoints, PNG previews, and training history are debug artifacts and
    # must not be counted as the compressed representation.
    return _runtime_directory_size(out_dir)

def _decode_bc1_payload(data: bytes, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Decode one standard BC1 mip into a [3,H,W] tensor."""
    blocks_y, blocks_x = (height + 3) // 4, (width + 3) // 4
    block_count = blocks_y * blocks_x
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size != block_count * 8:
        raise ValueError(f"BC1 payload size mismatch: {raw.size} != {block_count * 8}")
    raw = raw.reshape(block_count, 8).astype(np.uint32)
    ep0 = raw[:, 0] | (raw[:, 1] << 8)
    ep1 = raw[:, 2] | (raw[:, 3] << 8)
    selector_word = raw[:, 4] | (raw[:, 5] << 8) | (raw[:, 6] << 16) | (raw[:, 7] << 24)

    def rgb565(value):
        r = ((value >> 11) & 0x1F).astype(np.float32) / 31.0
        g = ((value >> 5) & 0x3F).astype(np.float32) / 63.0
        b = (value & 0x1F).astype(np.float32) / 31.0
        return np.stack([r, g, b], axis=-1)

    c0, c1 = rgb565(ep0), rgb565(ep1)
    palette_four = np.stack(
        [c0, c1, (2.0 * c0 + c1) / 3.0, (c0 + 2.0 * c1) / 3.0], axis=1
    )
    palette_three = np.stack(
        [c0, c1, (c0 + c1) / 2.0, np.zeros_like(c0)], axis=1
    )
    palette = np.where((ep0 > ep1)[:, None, None], palette_four, palette_three)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, :]
    selectors = ((selector_word[:, None] >> shifts) & 0x3).astype(np.int64)
    pixels = palette[np.arange(block_count)[:, None], selectors]
    pixels = pixels.reshape(blocks_y, blocks_x, 4, 4, 3)
    pixels = pixels.transpose(4, 0, 2, 1, 3).reshape(3, blocks_y * 4, blocks_x * 4)
    return torch.from_numpy(pixels[:, :height, :width].copy()).to(device=device, dtype=torch.float32)


def _read_bc1_dds(path: Path, mip_entries, device: torch.device):
    data = path.read_bytes()
    if len(data) < 128 or data[:4] != b"DDS ":
        raise ValueError(f"Invalid DDS file: {path}")
    offset = 128
    mips = []
    for entry in sorted(mip_entries, key=lambda item: int(item["mip_index"])):
        h, w = int(entry["shape_chw"][1]), int(entry["shape_chw"][2])
        size = int(entry["bytes"])
        payload = data[offset:offset + size]
        if len(payload) != size:
            raise ValueError(f"Truncated DDS mip in {path}")
        mips.append(_decode_bc1_payload(payload, h, w, device))
        offset += size
    return mips


class _RuntimeNeuralDecoder:
    """Decode exported BC1 latent textures and the FP16 MLP directly."""

    def __init__(self, out_dir: Path, metadata: dict, device: torch.device):
        self.device = device
        self.latents = []
        for latent_index in range(int(metadata["latent_count"])):
            entries = [
                item for item in metadata["latent_files"]
                if int(item["latent_index"]) == latent_index
            ]
            self.latents.append(
                _read_bc1_dds(out_dir / f"latent_{latent_index:02d}.bc1.dds", entries, device)
            )

        decoder_meta = metadata["decoder"]
        in_dim = int(decoder_meta["in_dim"])
        hidden = int(decoder_meta["hidden_dim"])
        out_dim = int(decoder_meta["out_dim"])
        blob = np.frombuffer(
            (out_dir / decoder_meta["weights_fp16_blob"]).read_bytes(), dtype="<f2"
        )
        expected = in_dim * hidden + hidden + hidden * hidden + hidden + out_dim * hidden + out_dim
        if blob.size != expected:
            raise ValueError(f"decoder_fp16.bin has {blob.size} values; expected {expected}")
        values = torch.from_numpy(blob.astype(np.float32, copy=True)).to(device)
        cursor = 0

        def take(count, shape):
            nonlocal cursor
            result = values[cursor:cursor + count].reshape(shape)
            cursor += count
            return result

        self.w1 = take(in_dim * hidden, (hidden, in_dim))
        self.b1 = take(hidden, (hidden,))
        self.w2 = take(hidden * hidden, (hidden, hidden))
        self.b2 = take(hidden, (hidden,))
        self.w3 = take(out_dim * hidden, (out_dim, hidden))
        self.b3 = take(out_dim, (out_dim,))
        self.max_lod = float(metadata.get("max_lod", 1.0))
        self.lod_biases = [float(value) for value in metadata["lod_biases"]]
        self.channel_mean = torch.tensor(metadata.get("channel_mean", [0.0] * out_dim), device=device)
        self.channel_std = torch.tensor(metadata.get("channel_std", [1.0] * out_dim), device=device)

    @torch.inference_mode()
    def forward(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        features = []
        for mips, bias in zip(self.latents, self.lod_biases):
            features.append(sample_mips_trilinear(mips, uv, lod + bias, bilinear_mode="bilinear"))
        x = torch.cat(features, dim=1)
        x = torch.cat([x, (lod / self.max_lod).unsqueeze(1)], dim=1)
        if x.shape[1] < self.w1.shape[1]:
            x = torch.nn.functional.pad(x, (0, self.w1.shape[1] - x.shape[1]))
        x = torch.nn.functional.silu(torch.nn.functional.linear(x, self.w1, self.b1))
        x = torch.nn.functional.silu(torch.nn.functional.linear(x, self.w2, self.b2))
        return torch.nn.functional.linear(x, self.w3, self.b3) * self.channel_std + self.channel_mean


# 使用导出的 BC1 latent 和 FP16 MLP 解码基础层级的完整纹理
@torch.inference_mode()
def _decode_base_texture(model, height: int, width: int, device, chunk_size: int):
    outputs = []
    pixel_count = height * width

    # 遍历所有像素
    for begin in range(0, pixel_count, chunk_size):
        end = min(begin + chunk_size, pixel_count)
        indices = torch.arange(begin, end, device=device)
        y = torch.div(indices, width, rounding_mode="floor")
        x = indices.remainder(width)
        uv = torch.stack(
            ((x.float() + 0.5) / width, (y.float() + 0.5) / height), dim=1
        )
        lod = torch.zeros(end - begin, dtype=torch.float32, device=device)

        # 解码当前分块
        outputs.append(model.forward(uv, lod).cpu())

     # 拼接分块，恢复纹理形状
    return torch.cat(outputs, dim=0).T.reshape(-1, height, width)

# 从神经压缩结果中解码 Gaussian 属性
def decompress_gaussians(out_dir: str | os.PathLike, config=None) -> Dict[str, np.ndarray]:
    config = dict(config or {})
    out_dir = Path(out_dir)

    # 读取压缩时保存的纹理尺寸、属性通道布局和归一化参数
    metadata = json.loads((out_dir / "gaussian_layout.json").read_text(encoding="utf-8"))

    # 获取解码设备，默认使用cuda
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    runtime_metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    model = _RuntimeNeuralDecoder(out_dir, runtime_metadata, device)

    # 解码完整的多通道基础纹理
    decoded = _decode_base_texture(
        model,
        metadata["height"],
        metadata["width"],
        device,
        int(config.get("decode_chunk_size", 262144)),
    )

    decoded = decoded.clamp(0.0, 1.0)
    result = {}

    # 根据元数据逐个恢复属性
    for name, info in metadata["attributes"].items():
        value = decoded[info["start"] : info["end"]]
        minimum = torch.tensor(info["minimum"]).reshape(-1, 1, 1)
        scale = torch.tensor(info["scale"]).reshape(-1, 1, 1)

        # 反归一化
        value = value * scale + minimum
        hwc = value.permute(1, 2, 0).contiguous().numpy()

        if name == "_rotation":
            norm = np.linalg.norm(hwc, axis=-1, keepdims=True)
            hwc = hwc / np.clip(norm, 1e-8, None)

        result[name] = hwc

    return result
