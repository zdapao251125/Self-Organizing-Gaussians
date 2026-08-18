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
    """Measure 2-D locality without sorting an enormous multi-channel tensor."""
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


# def _directory_size(path: Path) -> int:
#     return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

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
    names = {"decoder_fp16.bin", "metadata.json", "gaussian_layout.json"}
    files = [p for p in path.iterdir() if p.is_file() and (p.name in names or p.name.endswith(".bc1.dds"))]
    return sum(p.stat().st_size for p in files)

def compress_gaussians(gaussians, out_dir: str | os.PathLike, config=None) -> int:
    config = dict(config or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_offset = 0
    normalised_images = []
    layout = {}
    debug_original_grids = {}

    for name in _attribute_names(config):
        tensor = getattr(gaussians, name)
        _stats(f"gaussian/{name}", tensor)
        original_shape = list(tensor.shape[1:])
        grid = gaussians.attr_as_grid_img(name)
        height, width = int(grid.shape[0]), int(grid.shape[1])
        chw = grid.reshape(height, width, -1).permute(2, 0, 1).detach().float().cpu()
        _stats(f"attr_as_grid_img/{name}", chw)
        _grid_continuity(name, chw)
        normalised, minimum, scale = _normalise_chw(chw)
        _stats(f"normalization/{name}", normalised)
        _stats(f"normalization_min/{name}", minimum)
        _stats(f"normalization_scale/{name}", scale)
        _compare(f"normalization_roundtrip/{name}", normalised * scale + minimum, chw)
        debug_original_grids[name] = chw

        channels = int(chw.shape[0])
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

    reference = torch.cat(normalised_images, dim=0)
    _stats("reference_texture", reference)
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

    # Pass the per-channel inverse-normalisation metadata to training so losses
    # can be evaluated in the real Gaussian parameter domain as well as [0, 1].
    channel_minimum = []
    channel_scale = []
    for info in layout.values():
        channel_minimum.extend(info["minimum"])
        channel_scale.extend(info["scale"])
    config["channel_minimum"] = channel_minimum
    config["channel_scale"] = channel_scale

    # 训练神经纹理模型
    train_from_tensor(
        reference,
        out_dir,
        config,
    )

    # debug_reference.pt 已在 _directory_size() 中排除
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

    # train_from_tensor(reference, out_dir, config)
    return runtime_size


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
        value = value_raw.clamp(
            0.0,
            1.0,
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

    return result
    # result = {}
    # for name, info in metadata["attributes"].items():
    #     value = decoded[info["start"] : info["end"]]
    #     minimum = torch.tensor(info["minimum"]).reshape(-1, 1, 1)
    #     scale = torch.tensor(info["scale"]).reshape(-1, 1, 1)
    #     value = value * scale + minimum
    #     hwc = value.permute(1, 2, 0).contiguous().numpy()

    #     if name == "_rotation":
    #         norm = np.linalg.norm(hwc, axis=-1, keepdims=True)
    #         hwc = hwc / np.clip(norm, 1e-8, None)

    #     result[name] = hwc

    # return result
