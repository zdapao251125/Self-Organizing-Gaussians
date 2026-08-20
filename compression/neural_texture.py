from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression.neuralmaterialsBC1 import (  # noqa: E402
    NeuralMaterialCompressionModel,
    train_from_tensor,
)


DEFAULT_ATTRIBUTES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
)


def _attribute_names(config: Dict[str, Any]) -> Iterable[str]:
    attributes = config.get("attributes")
    if not attributes:
        return DEFAULT_ATTRIBUTES
    return [item["name"] if isinstance(item, dict) else item for item in attributes]


def _normalise_chw(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    minimum = image.amin(dim=(1, 2), keepdim=True)
    maximum = image.amax(dim=(1, 2), keepdim=True)
    scale = (maximum - minimum).clamp_min(1e-8)
    return (image - minimum) / scale, minimum, scale


def _normalise_chw_with_sparse_tails(
    image: torch.Tensor,
    tail_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robust per-channel normalization plus an exactly addressable sparse tail.

    Returns normalized/clipped data, low, scale, clipped data, flat CHW indices,
    and residual values. Quantiles are computed one channel at a time to avoid
    torch.quantile's large-input limitation.
    """
    if not 0.0 <= tail_fraction < 0.5:
        raise ValueError(f"tail_fraction must be in [0, 0.5), got {tail_fraction}")
    if tail_fraction == 0.0:
        normalized, minimum, scale = _normalise_chw(image)
        empty_i = torch.empty(0, dtype=torch.int64)
        empty_v = torch.empty(0, dtype=image.dtype)
        return normalized, minimum, scale, image, empty_i, empty_v

    flat = image.reshape(image.shape[0], -1)
    lows, highs = [], []
    for channel in range(flat.shape[0]):
        values = flat[channel]
        q = torch.quantile(
            values,
            values.new_tensor([tail_fraction, 1.0 - tail_fraction]),
        )
        lows.append(q[0])
        highs.append(q[1])
    minimum = torch.stack(lows).reshape(-1, 1, 1)
    maximum = torch.stack(highs).reshape(-1, 1, 1)
    scale = (maximum - minimum).clamp_min(1e-8)
    clipped = torch.maximum(torch.minimum(image, maximum), minimum)
    normalized = (clipped - minimum) / scale
    residual = image - clipped
    tail_mask = residual != 0
    tail_indices = tail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    tail_values = residual.reshape(-1)[tail_indices]
    return normalized, minimum, scale, clipped, tail_indices, tail_values


def _canonicalize_quaternion_chw(chw: torch.Tensor) -> torch.Tensor:
    """Choose one deterministic representative from equivalent q/-q pairs."""
    if chw.shape[0] != 4:
        raise ValueError(f"Expected four rotation channels, got {chw.shape[0]}")
    q = chw.permute(1, 2, 0).reshape(-1, 4)
    pivot = q.abs().argmax(dim=1, keepdim=True)
    pivot_value = q.gather(1, pivot)
    sign = torch.where(pivot_value < 0, -torch.ones_like(pivot_value), torch.ones_like(pivot_value))
    q = q * sign
    return q.reshape(chw.shape[1], chw.shape[2], 4).permute(2, 0, 1).contiguous()


def _stats(stage: str, value: torch.Tensor) -> None:
    x = value.detach().float().cpu()
    finite = torch.isfinite(x)
    print(
        f"[stage:{stage}] shape={tuple(x.shape)} dtype={x.dtype} "
        f"finite={finite.float().mean().item():.6f} "
        f"nan={torch.isnan(x).sum().item()} inf={torch.isinf(x).sum().item()}"
    )
    x = x[finite]
    if x.numel():
        print(
            f"[stage:{stage}] min={x.min().item():.7g} max={x.max().item():.7g} "
            f"mean={x.mean().item():.7g} std={x.std(unbiased=False).item():.7g}"
        )


def _compare(stage: str, pred: torch.Tensor, target: torch.Tensor, peak: float | None = None) -> None:
    p, t = pred.detach().float().cpu(), target.detach().float().cpu()
    if p.shape != t.shape:
        print(f"[compare:{stage}] SHAPE MISMATCH pred={tuple(p.shape)} target={tuple(t.shape)}")
        return
    valid = torch.isfinite(p) & torch.isfinite(t)
    p, t = p[valid], t[valid]
    if not p.numel():
        print(f"[compare:{stage}] no finite samples")
        return
    e = p - t
    mse = e.square().mean()
    rmse = mse.sqrt()
    dynamic_range = float(peak) if peak is not None else max((t.max() - t.min()).item(), 1e-8)
    psnr = 20.0 * np.log10(dynamic_range / max(rmse.item(), 1e-8))
    print(
        f"[compare:{stage}] mae={e.abs().mean().item():.7g} rmse={rmse.item():.7g} "
        f"max_abs={e.abs().max().item():.7g} psnr={psnr:.3f}dB peak={dynamic_range:.7g}"
    )


def _error_quantiles_and_outliers(
    name: str,
    pred_chw: torch.Tensor,
    target_chw: torch.Tensor,
    original_grids: dict,
    topk: int = 10,
) -> None:
    """Report per-Gaussian tails and exact grid indices for severe errors."""
    error = (pred_chw.float() - target_chw.float()).square().mean(0).sqrt()
    q = torch.quantile(error.reshape(-1), torch.tensor([.5, .95, .99, .999]))
    print(
        f"[quantile:final_raw/{name}] p50={q[0].item():.7g} "
        f"p95={q[1].item():.7g} p99={q[2].item():.7g} "
        f"p99.9={q[3].item():.7g} max={error.max().item():.7g}"
    )
    count = min(int(topk), error.numel())
    values, indices = torch.topk(error.reshape(-1), count)
    width = error.shape[1]
    opacity = original_grids.get("_opacity")
    for rank, (err, flat) in enumerate(zip(values, indices), 1):
        flat_i = int(flat.item())
        y, x = divmod(flat_i, width)
        opacity_raw = float(opacity[0, y, x].item()) if opacity is not None else float("nan")
        opacity_active = 1.0 / (1.0 + np.exp(-opacity_raw))
        pred_values = pred_chw[:, y, x].tolist()
        target_values = target_chw[:, y, x].tolist()
        print(
            f"[outlier:{name}] rank={rank} grid=({y},{x}) flat_index={flat_i} "
            f"rms={err.item():.7g} opacity_logit={opacity_raw:.6g} "
            f"opacity_sigmoid={opacity_active:.6g} pred={pred_values} target={target_values}"
        )


def _grid_continuity(name: str, chw: torch.Tensor, max_quantile_samples: int = 1_000_000) -> None:
    """Measure 2-D locality without sorting an enormous multi-channel tensor.

    The mean uses every horizontal/vertical neighbour. Quantiles use a
    deterministic strided sample capped at ``max_quantile_samples``; otherwise
    45 SH channels at a 1264-square grid exceed torch.quantile's input limit.
    """
    per_direction = max(1, int(max_quantile_samples) // 2)

    def _strided_sample(value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(-1)
        stride = max(1, math.ceil(flat.numel() / per_direction))
        return flat[::stride][:per_direction]

    dx = (chw[:, :, 1:] - chw[:, :, :-1]).abs()
    dx_count, dx_sum = dx.numel(), dx.sum()
    dx_sample = _strided_sample(dx)
    del dx
    dy = (chw[:, 1:, :] - chw[:, :-1, :]).abs()
    dy_count, dy_sum = dy.numel(), dy.sum()
    dy_sample = _strided_sample(dy)
    del dy
    count = dx_count + dy_count
    local_mean = (dx_sum + dy_sum) / max(count, 1)
    sample = torch.cat((dx_sample, dy_sample))
    baseline = chw.std(dim=(1, 2), unbiased=False).mean().clamp_min(1e-8)
    q = torch.quantile(
        sample,
        torch.tensor([.5, .95, .99, .999], device=sample.device, dtype=sample.dtype),
    )
    ratio = local_mean / baseline
    print(
        f"[continuity:{name}] neighbor_mae={local_mean.item():.7g} "
        f"neighbor_mae/std={ratio.item():.4f} p50={q[0].item():.7g} "
        f"p95={q[1].item():.7g} p99={q[2].item():.7g} p99.9={q[3].item():.7g} "
        f"quantile_samples={sample.numel()}/{count}"
    )

def _directory_size(path: Path) -> int:

    ignored_names = {
        "debug_reference.pt",
        "debug_original_grids.pt",
    }

    total_size = 0

    for file in path.rglob("*"):
        if not file.is_file():
            continue

        if file.name in ignored_names:
            continue

        total_size += file.stat().st_size

    return total_size

def _runtime_artifact_size(path: Path) -> int:
    """Size of intended deployment files, excluding checkpoints/debug previews."""
    names = {
        "decoder_fp16.bin", "metadata.json", "gaussian_layout.json",
        "tail_residuals.npz", "prediction_corrections.npz",
        "decoder_features_rest_fp16.bin", "decoder_scaling_fp16.bin",
    }
    files = [p for p in path.iterdir() if p.is_file() and (p.name in names or p.name.endswith(".bc1.dds"))]
    return sum(p.stat().st_size for p in files)

@torch.no_grad()
def _save_prediction_error_corrections(
    model, reference, original_grids, layout, out_dir, config
):
    """Store exact vectors for the worst *post-training* Gaussian errors.

    Source-value tails and prediction-error tails are different sets. This
    second sparse layer is selected only after quantized decoding, and stores
    complete attribute vectors per spatial Gaussian to avoid inconsistent
    component-wise repairs.
    """
    fraction = float(config.get("prediction_correction_fraction", 0.002))
    if fraction <= 0.0:
        return
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    decoded = _decode_base_texture(
        model, int(reference.shape[1]), int(reference.shape[2]), device,
        int(config.get("decode_chunk_size", 262144)),
    )
    dtype_name = str(config.get("prediction_correction_dtype", "float16")).lower()
    if dtype_name not in ("float16", "float32"):
        raise ValueError("prediction_correction_dtype must be float16 or float32")
    np_dtype = np.float16 if dtype_name == "float16" else np.float32
    per_attr = config.get("prediction_correction_fractions", {}) or {}
    opacity = original_grids.get("_opacity")
    visibility = None if opacity is None else torch.sigmoid(opacity.float()).squeeze(0)
    archive = {}
    tail_path = Path(out_dir) / "tail_residuals.npz"
    tail_archive = np.load(tail_path, allow_pickle=False) if tail_path.exists() else None
    total = int(reference.shape[1] * reference.shape[2])
    for attr_id, (name, info) in enumerate(layout.items()):
        attr_fraction = float(per_attr.get(name, fraction))
        if attr_fraction <= 0.0:
            continue
        start, end = int(info["start"]), int(info["end"])
        minimum = torch.tensor(info["minimum"]).reshape(-1, 1, 1)
        scale = torch.tensor(info["scale"]).reshape(-1, 1, 1)
        normalized = decoded[start:end].cpu()
        if name != "_rotation":
            normalized = normalized.clamp(0.0, 1.0)
        reconstructed = normalized * scale + minimum
        original = original_grids[name].float()
        # 在现有源尾部覆盖后对管道进行评分，否则新存档将浪费空间，因为需要修正已存储的值。
        tail_key = info.get("tail_key")
        if tail_archive is not None and tail_key and int(info.get("tail_count", 0)):
            scalar_indices = torch.from_numpy(
                tail_archive[f"{tail_key}_indices"].astype(np.int64, copy=False)
            )
            tail_values = torch.from_numpy(
                tail_archive[f"{tail_key}_residuals"].astype(np.float32, copy=False)
            )
            reconstructed.reshape(-1)[scalar_indices] = tail_values
        if name == "_rotation":
            q0 = torch.nn.functional.normalize(reconstructed, dim=0, eps=1e-8)
            q1 = torch.nn.functional.normalize(original, dim=0, eps=1e-8)
            score = 2.0 * torch.acos((q0 * q1).sum(0).abs().clamp(0, 1 - 1e-7))
        else:
            score = (reconstructed - original).square().mean(0).sqrt()
        # 优先处理可见高斯分布的几何/形状参数中的误差。
        if visibility is not None and name in ("_xyz", "_scaling", "_rotation"):
            score = score * (0.1 + 0.9 * visibility)
        count = min(total, max(1, int(round(total * attr_fraction))))
        indices = torch.topk(score.reshape(-1), count, largest=True, sorted=False).indices
        values = original.reshape(original.shape[0], -1)[:, indices].T.contiguous()
        key = f"attr_{attr_id}"
        archive[f"{key}_indices"] = indices.numpy().astype(np.uint32)
        archive[f"{key}_values"] = values.numpy().astype(np_dtype)
        info["prediction_correction_key"] = key
        info["prediction_correction_count"] = count
        info["prediction_correction_fraction"] = attr_fraction
        print(
            f"[prediction-correction:{name}] vectors={count}/{total} "
            f"fraction={count / total:.4%} score_min={score.reshape(-1)[indices].min().item():.6g}"
        )
    if archive:
        path = Path(out_dir) / "prediction_corrections.npz"
        np.savez_compressed(path, **archive)
        print(f"[prediction-correction] saved {path} ({path.stat().st_size / 1024**2:.3f} MiB)")
    if tail_archive is not None:
        tail_archive.close()
    if was_training:
        model.train()


def compress_gaussians(gaussians, out_dir: str | os.PathLike, config=None) -> int:
    config = dict(config or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_offset = 0
    normalised_images = []
    layout = {}
    debug_original_grids = {}
    sparse_tail_arrays = {}
    tail_fraction = float(config.get("tail_fraction", 0.001))
    tail_dtype_name = str(config.get("tail_residual_dtype", "float16")).lower()
    if tail_dtype_name not in ("float16", "float32"):
        raise ValueError("tail_residual_dtype must be 'float16' or 'float32'")
    tail_numpy_dtype = np.float16 if tail_dtype_name == "float16" else np.float32
    tail_attributes_cfg = config.get("tail_attributes")
    tail_attributes = set(tail_attributes_cfg) if tail_attributes_cfg else set(_attribute_names(config))

    for name in _attribute_names(config):
        tensor = getattr(gaussians, name)
        _stats(f"gaussian/{name}", tensor)
        original_shape = list(tensor.shape[1:])
        grid = gaussians.attr_as_grid_img(name)
        height, width = int(grid.shape[0]), int(grid.shape[1])
        chw = grid.reshape(height, width, -1).permute(2, 0, 1).detach().float().cpu()
        if name == "_rotation" and bool(config.get("canonicalize_rotation_sign", True)):
            chw = _canonicalize_quaternion_chw(chw)
            print("[rotation] canonicalized q/-q sign using the largest-magnitude component")
        _stats(f"attr_as_grid_img/{name}", chw)
        _grid_continuity(name, chw)
        attr_tail_fraction = tail_fraction if name in tail_attributes else 0.0
        normalised, minimum, scale, clipped, tail_indices, tail_values = (
            _normalise_chw_with_sparse_tails(chw, attr_tail_fraction)
        )
        _stats(f"normalization/{name}", normalised)
        _stats(f"normalization_min/{name}", minimum)
        _stats(f"normalization_scale/{name}", scale)
        _compare(f"normalization_core_roundtrip/{name}", normalised * scale + minimum, clipped)
        tail_key = f"attr_{len(layout)}"
        # 存储绝对原始值，而非原始值减去裁剪后的残差。
        # 添加源残差时假设神经解码器准确复现了裁剪边界；即使解码器在尾部位置不准确，绝对覆盖仍保持正确。
        original_tail_values = chw.reshape(-1)[tail_indices]
        stored_tail_values_np = original_tail_values.cpu().numpy().astype(tail_numpy_dtype)
        stored_tail_values = torch.from_numpy(
            stored_tail_values_np.astype(np.float32, copy=False)
        ).to(chw.dtype)
        sparse_tail_arrays[f"{tail_key}_indices"] = tail_indices.cpu().numpy().astype(np.uint32)
        sparse_tail_arrays[f"{tail_key}_residuals"] = stored_tail_values_np
        restored_with_tail = (normalised * scale + minimum).reshape(-1)
        if tail_indices.numel():
            restored_with_tail = restored_with_tail.clone()
            restored_with_tail[tail_indices] = stored_tail_values
        _compare(
            f"normalization_plus_sparse_tail_roundtrip/{name}",
            restored_with_tail.reshape_as(chw), chw,
        )
        print(
            f"[tail:{name}] fraction_each_side={attr_tail_fraction:.6g} "
            f"elements={tail_indices.numel()}/{chw.numel()} "
            f"ratio={tail_indices.numel() / max(chw.numel(), 1):.6%} "
            f"residual_dtype={tail_dtype_name}"
        )
        debug_original_grids[name] = chw

        channels = int(chw.shape[0])
        layout[name] = {
            "start": channel_offset,
            "end": channel_offset + channels,
            "tensor_tail_shape": original_shape,
            "minimum": minimum.flatten().tolist(),
            "scale": scale.flatten().tolist(),
            "tail_key": tail_key,
            "tail_count": int(tail_indices.numel()),
            "tail_fraction_each_side": attr_tail_fraction,
            "tail_residual_dtype": tail_dtype_name,
            "tail_storage": "absolute_values_v1",
        }
        channel_offset += channels
        normalised_images.append(normalised)

    if not normalised_images:
        raise ValueError("Neural texture compression needs at least one attribute.")

    reference = torch.cat(normalised_images, dim=0)
    _stats("reference_texture", reference)
    metadata = {
        "format_version": 2,
        "height": int(reference.shape[1]),
        "width": int(reference.shape[2]),
        "channels": int(reference.shape[0]),
        "attributes": layout,
        "sparse_tail_file": "tail_residuals.npz",
    }
    (out_dir / "gaussian_layout.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    np.savez_compressed(out_dir / "tail_residuals.npz", **sparse_tail_arrays)
    print(
        "[tail] sparse residual file: "
        f"{(out_dir / 'tail_residuals.npz').stat().st_size / 1024**2:.3f} MiB"
    )

    # -------------------------------------------------
    # 调试数据：保存训练使用的归一化参考纹理。
    #
    # 该文件只用于计算属性重建误差，不能作为压缩结果
    # 计入压缩大小，也不是运行时解码所需文件。
    # -------------------------------------------------

    print(
        "[neural-texture] reference shape:",
        tuple(reference.shape),
    )

    debug_reference_path = (
        out_dir
        / "debug_reference.pt"
    )

    save_debug_reference = bool(
        config.get(
            "save_debug_reference",
            True,
        )
    )

    if save_debug_reference:
        torch.save(
            reference.cpu(),
            debug_reference_path,
        )

        print(
            "[debug] reference saved:",
            debug_reference_path,
        )

        print(
            "[debug] reference size: "
            f"{debug_reference_path.stat().st_size / 1024**2:.2f} MiB"
        )
        torch.save(debug_original_grids, out_dir / "debug_original_grids.pt")

    # 将每通道的逆归一化元数据传递给训练，以便损失函数既可以在真实的高斯参数域中评估，也可以在[0,1]范围内进行评估。
    channel_minimum = []
    channel_scale = []
    for info in layout.values():
        channel_minimum.extend(info["minimum"])
        channel_scale.extend(info["scale"])
    config["channel_minimum"] = channel_minimum
    config["channel_scale"] = channel_scale

    # 训练神经纹理模型
    model, _history = train_from_tensor(
        reference,
        out_dir,
        config,
    )
    _save_prediction_error_corrections(
        model, reference, debug_original_grids, layout, out_dir, config
    )
    metadata["prediction_correction_file"] = "prediction_corrections.npz"
    metadata["prediction_correction_storage"] = "absolute_vectors_v1"
    (out_dir / "gaussian_layout.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    total_size = _directory_size(out_dir)
    runtime_size = _runtime_artifact_size(out_dir)

    print(
        "[neural-texture] counted size: "
        f"{total_size / 1024**2:.2f} MiB"
    )
    print(
        "[neural-texture] deployment artifacts only: "
        f"{runtime_size / 1024**2:.2f} MiB "
        "(DDS + FP16 decoder + runtime metadata)"
    )

    return runtime_size


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    valid_keys = {
        "latent_resolutions",
        "latent_mips",
        "out_channels",
        "hidden_dim",
        "scaling_decoder_hidden_dim",
        "features_rest_decoder_hidden_dim",
        "separate_gaussian_decoders",
        "ref_base_res",
        "max_lod",
    }
    model_config = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in valid_keys
    }
    if "separate_gaussian_decoders" not in model_config:
        model_config["separate_gaussian_decoders"] = any(
            key.startswith("features_rest_decoder.")
            for key in checkpoint["model_state_dict"]
        )
    model = NeuralMaterialCompressionModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_freeze_bc_features(True)
    model.eval()
    return model


@torch.inference_mode()
def _decode_base_texture(model, height: int, width: int, device, chunk_size: int):
    outputs = []
    pixel_count = height * width

    for begin in range(0, pixel_count, chunk_size):
        end = min(begin + chunk_size, pixel_count)
        indices = torch.arange(begin, end, device=device)
        y = torch.div(indices, width, rounding_mode="floor")
        x = indices.remainder(width)
        uv = torch.stack(
            ((x.float() + 0.5) / width, (y.float() + 0.5) / height), dim=1
        )
        lod = torch.zeros(end - begin, dtype=torch.float32, device=device)
        outputs.append(model.forward_bc(uv, lod).cpu())

    return torch.cat(outputs, dim=0).T.reshape(-1, height, width)


def decompress_gaussians(out_dir: str | os.PathLike, config=None) -> Dict[str, np.ndarray]:
    config = dict(config or {})
    out_dir = Path(out_dir)
    metadata = json.loads((out_dir / "gaussian_layout.json").read_text(encoding="utf-8"))

    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print(
        "[evaluation-mode] Decoding checkpoint.pt. This validates the quantized "
        "PyTorch path, but not yet the exported DDS reader or FP16 decoder runtime."
    )
    model = _load_model(out_dir / "checkpoint.pt", device)
    decoded_raw = _decode_base_texture(
        model,
        metadata["height"],
        metadata["width"],
        device,
        int(config.get("decode_chunk_size", 262144)),
    )
    # decoded = decoded.clamp(0.0, 1.0)
# -------------------------------------------------
    # 加载调试参考纹理
    # -------------------------------------------------
    debug_reference_path = (
        out_dir
        / "debug_reference.pt"
    )

    debug_reference = None
    debug_original_grids = None
    tail_archive = None
    correction_archive = None

    if debug_reference_path.exists():
        debug_reference = torch.load(
            debug_reference_path,
            map_location="cpu",
        )

        debug_reference = (
            debug_reference
            .detach()
            .float()
            .contiguous()
        )

        print(
            "[debug] loaded reference:",
            debug_reference_path,
        )

        print(
            "[debug] reference shape:",
            tuple(debug_reference.shape),
        )

        if (
            tuple(debug_reference.shape)
            != tuple(decoded_raw.shape)
        ):
            raise ValueError(
                "Debug reference shape mismatch: "
                f"reference={tuple(debug_reference.shape)}, "
                f"decoded={tuple(decoded_raw.shape)}"
            )
    else:
        print(
            "[debug] debug_reference.pt not found; "
            "per-attribute MAE/RMSE/PSNR "
            "will not be calculated."
        )

    original_grids_path = out_dir / "debug_original_grids.pt"
    if original_grids_path.exists():
        debug_original_grids = torch.load(original_grids_path, map_location="cpu")
    tail_path = out_dir / "tail_residuals.npz"
    if tail_path.exists():
        tail_archive = np.load(tail_path, allow_pickle=False)
        print(f"[tail] loaded sparse residuals: {tail_path}")
    correction_path = out_dir / metadata.get(
        "prediction_correction_file", "prediction_corrections.npz"
    )
    if correction_path.exists():
        correction_archive = np.load(correction_path, allow_pickle=False)
        print(f"[prediction-correction] loaded: {correction_path}")

    result = {}

    # 用于计算所有通道的整体指标
    all_squared_error_sum = 0.0
    all_absolute_error_sum = 0.0
    all_value_count = 0

    for (
        name,
        info,
    ) in metadata["attributes"].items():
        start = int(
            info["start"]
        )
        end = int(
            info["end"]
        )

        # 当前属性的归一化预测值，尚未 clamp
        value_raw = (
            decoded_raw[start:end]
            .detach()
            .float()
            .cpu()
        )
        _stats(f"decoder_raw/{name}", value_raw)

        # -------------------------------------------------
        # 检查预测值是否超出训练目标的 [0,1] 范围
        # -------------------------------------------------
        outside_mask = (
            (value_raw < 0.0)
            | (value_raw > 1.0)
        )

        outside_ratio = (
            outside_mask
            .float()
            .mean()
            .item()
        )

        below_ratio = (
            (value_raw < 0.0)
            .float()
            .mean()
            .item()
        )

        above_ratio = (
            (value_raw > 1.0)
            .float()
            .mean()
            .item()
        )

        print(
            f"[decode] {name}: "
            f"normalized_min="
            f"{value_raw.min().item():.6f}, "
            f"normalized_max="
            f"{value_raw.max().item():.6f}, "
            f"outside="
            f"{outside_ratio * 100:.4f}%, "
            f"below="
            f"{below_ratio * 100:.4f}%, "
            f"above="
            f"{above_ratio * 100:.4f}%"
        )

        # -------------------------------------------------
        # 和训练时保存的归一化参考纹理比较
        # -------------------------------------------------
        if debug_reference is not None:
            target = (
                debug_reference[start:end]
                .detach()
                .float()
                .cpu()
            )

            error = (
                value_raw
                - target
            )

            absolute_error = (
                error.abs()
            )

            squared_error = (
                error.square()
            )

            mae = (
                absolute_error
                .mean()
                .item()
            )

            rmse = (
                squared_error
                .mean()
                .sqrt()
                .item()
            )

            # 因为参考值归一化在 [0,1]，
            # MAX_I = 1，所以 PSNR = -20 log10(RMSE)
            if rmse > 0.0:
                normalized_psnr = (
                    -20.0
                    * np.log10(rmse)
                )
            else:
                normalized_psnr = (
                    float("inf")
                )

            max_absolute_error = (
                absolute_error
                .max()
                .item()
            )

            print(
                f"[error] {name}: "
                f"mae={mae:.6f}, "
                f"rmse={rmse:.6f}, "
                f"max_abs="
                f"{max_absolute_error:.6f}, "
                f"normalized_psnr="
                f"{normalized_psnr:.2f} dB"
            )
            _compare(f"decoder_raw_vs_reference/{name}", value_raw, target, peak=1.0)

            all_absolute_error_sum += (
                absolute_error.sum().item()
            )

            all_squared_error_sum += (
                squared_error.sum().item()
            )

            all_value_count += (
                error.numel()
            )

        # -------------------------------------------------
        # 将预测值限制到训练目标范围
        # -------------------------------------------------
        # 按组件分量夹紧会改变四元数的方向。旋转首先被去归一化，然后投影到下方的单位球面上。
        value = (
            value_raw
            if name == "_rotation" and not bool(
                config.get("clamp_rotation_before_denormalization", False)
            )
            else value_raw.clamp(0.0, 1.0)
        )
        _stats(f"clamp/{name}", value)
        _compare(f"clamp_change/{name}", value, value_raw, peak=1.0)
        if debug_reference is not None:
            _compare(f"clamped_vs_reference/{name}", value, target, peak=1.0)

        minimum = torch.tensor(
            info["minimum"],
            dtype=value.dtype,
        ).reshape(
            -1,
            1,
            1,
        )

        scale = torch.tensor(
            info["scale"],
            dtype=value.dtype,
        ).reshape(
            -1,
            1,
            1,
        )

        # 从 [0,1] 恢复到原始 Gaussian 参数范围
        value = (
            value * scale
            + minimum
        )
        tail_key = info.get("tail_key")
        tail_count = int(info.get("tail_count", 0))
        if tail_archive is not None and tail_key and tail_count:
            index_key = f"{tail_key}_indices"
            value_key = f"{tail_key}_residuals"
            indices = torch.from_numpy(tail_archive[index_key].astype(np.int64, copy=False))
            stored_values = torch.from_numpy(
                tail_archive[value_key].astype(np.float32, copy=False)
            ).to(value.dtype)
            value_flat = value.reshape(-1)
            if info.get("tail_storage") == "absolute_values_v1":
                value_flat[indices] = stored_values
                action = "overwritten"
            else:
                # 与包含原始减去裁剪残差的旧存档实现向后兼容。
                value_flat[indices] += stored_values
                action = "residual-added (legacy)"
            print(
                f"[tail:{name}] {action}={indices.numel()} elements "
                f"stored_abs_max={stored_values.abs().max().item():.7g}"
            )
        correction_key = info.get("prediction_correction_key")
        correction_count = int(info.get("prediction_correction_count", 0))
        if correction_archive is not None and correction_key and correction_count:
            indices = torch.from_numpy(
                correction_archive[f"{correction_key}_indices"].astype(np.int64, copy=False)
            )
            vectors = torch.from_numpy(
                correction_archive[f"{correction_key}_values"].astype(np.float32, copy=False)
            ).to(value.dtype)
            # 归档布局为[K,C]，而纹理视图为[C,H*W]。
            value.reshape(value.shape[0], -1)[:, indices] = vectors.T
            print(
                f"[prediction-correction:{name}] overwritten={indices.numel()} vectors"
            )
        _stats(f"denormalization/{name}", value)
        if debug_original_grids is not None and name in debug_original_grids:
            _compare(f"denormalized_vs_original_grid/{name}", value, debug_original_grids[name])
            if name in ("_xyz", "_scaling"):
                _error_quantiles_and_outliers(
                    name, value, debug_original_grids[name], debug_original_grids,
                    topk=int(config.get("debug_outlier_topk", 10)),
                )

        print(
            f"[decode] {name}: "
            f"restored_min="
            f"{value.min().item():.6f}, "
            f"restored_max="
            f"{value.max().item():.6f}"
        )

        # [C,H,W] → [H,W,C]
        hwc = (
            value
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )

        # 恢复 rotation 后保证四元数长度为1
        if name == "_rotation":
            rotation_before = torch.from_numpy(hwc.copy())
            norm = np.linalg.norm(
                hwc,
                axis=-1,
                keepdims=True,
            )

            hwc = hwc / np.clip(
                norm,
                1e-8,
                None,
            )
            rotation_after = torch.from_numpy(hwc.copy())
            _stats("rotation/norm_before", torch.from_numpy(norm))
            _stats("rotation/normalized", rotation_after)
            _compare("rotation/normalization_change", rotation_after, rotation_before)
            if debug_original_grids is not None and name in debug_original_grids:
                original_hwc = debug_original_grids[name].permute(1, 2, 0)
                original_unit = torch.nn.functional.normalize(original_hwc, dim=-1, eps=1e-8)
                _compare("rotation/final_vs_original_unit", rotation_after, original_unit)
                dot = (rotation_after * original_unit).sum(-1).abs().clamp(0, 1 - 1e-7)
                angle_deg = 2.0 * torch.acos(dot) * (180.0 / np.pi)
                aq = torch.quantile(angle_deg.reshape(-1), torch.tensor([.5, .95, .99, .999]))
                print(
                    f"[rotation-angle:final] p50={aq[0].item():.3f}deg "
                    f"p95={aq[1].item():.3f}deg p99={aq[2].item():.3f}deg "
                    f"p99.9={aq[3].item():.3f}deg max={angle_deg.max().item():.3f}deg"
                )

        result[name] = hwc
        _stats(f"reconstructed_gaussian/{name}", torch.from_numpy(hwc))

    # -------------------------------------------------
    # 所有通道合并后的总体误差
    # -------------------------------------------------
    if (
        debug_reference is not None
        and all_value_count > 0
    ):
        overall_mae = (
            all_absolute_error_sum
            / all_value_count
        )

        overall_rmse = (
            all_squared_error_sum
            / all_value_count
        ) ** 0.5

        if overall_rmse > 0.0:
            overall_psnr = (
                -20.0
                * np.log10(
                    overall_rmse
                )
            )
        else:
            overall_psnr = (
                float("inf")
            )

        print(
            "[error] overall normalized: "
            f"mae={overall_mae:.6f}, "
            f"rmse={overall_rmse:.6f}, "
            f"psnr={overall_psnr:.2f} dB"
        )

    if tail_archive is not None:
        tail_archive.close()
    if correction_archive is not None:
        correction_archive.close()
    return result
