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
    NeuralMaterialCompressionModel,
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

    # 返回整个输出目录的文件总大小，统计压缩结果体积
    return _directory_size(out_dir)

# 加载模型
def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    valid_keys = {
        "latent_resolutions",
        "latent_mips",
        "out_channels",
        "hidden_dim",
        "ref_base_res",
        "max_lod",
    }
    
    model_config = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in valid_keys
    }
    model = NeuralMaterialCompressionModel(**model_config).to(device)

    # 加载训练后保存的模型参数
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_freeze_bc_features(True)
    model.eval()
    return model

# 使用神经压缩模型解码基础层级的完整纹理
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
        outputs.append(model.forward_bc(uv, lod).cpu())

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

    # 加载神经压缩模型及训练权重
    model = _load_model(out_dir / "checkpoint.pt", device)

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