import sys
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import get_hydra_training_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

import yaml
from dataclasses import dataclass, asdict
import pandas as pd

from compression.jpeg_xl import JpegXlCodec
from compression.npz import NpzCodec
from compression.exr import EXRCodec
from compression.png import PNGCodec
from compression.neural_texture import compress_gaussians, decompress_gaussians

codecs = {
    "jpeg-xl": JpegXlCodec,
    "npz": NpzCodec,
    "exr": EXRCodec,
    "png": PNGCodec,
}

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


GAUSSIAN_ATTRIBUTE_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
)


def _copy_gaussian_template(dst, src):
    """Seed a decoded model with attributes intentionally left uncompressed."""
    if src is None:
        return
    for attr_name in GAUSSIAN_ATTRIBUTE_NAMES:
        value = getattr(src, attr_name, None)
        if isinstance(value, torch.Tensor) and value.numel():
            setattr(dst, attr_name, value.detach().clone())
    if hasattr(src, "grid_sidelen"):
        dst.grid_sidelen = src.grid_sidelen


def _attribute_name(attribute):
    return attribute["name"] if isinstance(attribute, dict) else attribute


def _validate_hybrid_attributes(neural_attributes, sog_attributes):
    """Require a standalone stream with every Gaussian attribute exactly once."""
    neural_names = [_attribute_name(item) for item in neural_attributes]
    sog_names = [_attribute_name(item) for item in sog_attributes]
    overlap = sorted(set(neural_names) & set(sog_names))
    missing = sorted(set(GAUSSIAN_ATTRIBUTE_NAMES) - set(neural_names) - set(sog_names))
    unknown = sorted((set(neural_names) | set(sog_names)) - set(GAUSSIAN_ATTRIBUTE_NAMES))
    duplicates = sorted(
        name for name in set(neural_names + sog_names)
        if (neural_names + sog_names).count(name) > 1
    )
    if overlap or missing or unknown or duplicates:
        raise ValueError(
            "Invalid hybrid attribute layout: "
            f"overlap={overlap}, missing={missing}, unknown={unknown}, "
            f"duplicates={duplicates}"
        )
    return neural_names, sog_names


def _morton_code_3d(integer_xyz, bits):
    code = torch.zeros(
        integer_xyz.shape[0], dtype=torch.int64, device=integer_xyz.device
    )
    for bit in range(int(bits)):
        code |= ((integer_xyz[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((integer_xyz[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((integer_xyz[:, 2] >> bit) & 1) << (3 * bit + 2)
    return code


@torch.no_grad()
def _prepare_morton_block_layout(gaussians, experiment_config):
    """Build the XYZ Morton/tile order without changing the source layout.

    The returned array maps Morton-layout positions to original SOG/PLAS
    positions. Keeping the source model untouched lets the non-XYZ stream use
    the original layout while XYZ uses the spatially local layout.
    """
    params = dict(experiment_config.get('params', {}) or {})
    if not bool(params.get('xyz_morton_block_sort', False)):
        return None
    tile_size = int(params.get('xyz_block_size', 16))
    morton_bits = int(params.get('xyz_morton_bits', 10))
    count = int(gaussians._xyz.shape[0])
    side = int(np.sqrt(count))
    if side * side != count:
        raise ValueError('prune_to_square_shape() must run before Morton layout')
    if tile_size <= 0 or side % tile_size != 0:
        raise ValueError(
            f'Grid side {side} must be divisible by xyz_block_size={tile_size}'
        )

    xyz = gaussians._xyz.detach().float().reshape(count, -1)[:, :3]
    sample = xyz
    if count > 200000:
        sample_ids = torch.linspace(0, count - 1, 200000, device=xyz.device).long()
        sample = xyz[sample_ids]
    low = torch.quantile(sample, 0.001, dim=0)
    high = torch.quantile(sample, 0.999, dim=0)
    xyz01 = ((xyz - low) / (high - low).clamp_min(1e-8)).clamp(0.0, 1.0)
    integer_xyz = torch.round(xyz01 * ((1 << morton_bits) - 1)).long()
    spatial_order = torch.argsort(
        _morton_code_3d(integer_xyz, morton_bits), stable=True
    )

    tiles_per_side = side // tile_size
    final_order = (
        spatial_order.reshape(tiles_per_side, tiles_per_side, tile_size, tile_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )
    print(
        f'[xyz-layout] generated Morton({morton_bits} bit/axis) order -> '
        f'{tiles_per_side}x{tiles_per_side} tiles -> '
        f'{side}x{side} grid, tile={tile_size}x{tile_size}; '
        'non-XYZ attributes keep the original SOG/PLAS order'
    )
    return final_order


def _clone_gaussians(gaussians):
    """Create an attribute-only clone suitable for compression layout changes."""
    cloned = GaussianModel(
        gaussians.max_sh_degree, gaussians.disable_xyz_log_activation
    )
    cloned.active_sh_degree = gaussians.active_sh_degree
    _copy_gaussian_template(cloned, gaussians)
    return cloned


def _restore_grid_from_layout(grid, layout_to_original):
    """Restore decoded XYZ from Morton layout to the original SOG/PLAS order.

    ``decompress_gaussians`` currently returns NumPy HWC arrays, while some
    callers may use torch CHW/HWC tensors. Support all of those representations
    and preserve the input type and layout.
    """
    if grid.ndim != 3:
        raise ValueError(f'Expected a 3D decoded XYZ grid, got shape={tuple(grid.shape)}')

    is_tensor = isinstance(grid, torch.Tensor)
    order_count = int(layout_to_original.numel())
    is_hwc = grid.shape[-1] == 3 and grid.shape[0] * grid.shape[1] == order_count
    is_chw = grid.shape[0] == 3 and grid.shape[1] * grid.shape[2] == order_count
    if not is_hwc and not is_chw:
        raise ValueError(
            'Morton index length/layout does not match decoded XYZ grid: '
            f'indices={order_count}, grid={tuple(grid.shape)}'
        )

    if is_tensor:
        order = layout_to_original.to(device=grid.device, dtype=torch.long)
        records = (
            grid.reshape(-1, 3)
            if is_hwc
            else grid.permute(1, 2, 0).contiguous().reshape(-1, 3)
        )
        restored = torch.empty_like(records)
        restored[order] = records
        if is_hwc:
            return restored.reshape(grid.shape)
        return restored.reshape(grid.shape[1], grid.shape[2], 3).permute(2, 0, 1).contiguous()

    order = layout_to_original.cpu().numpy().astype(np.int64, copy=False)
    records = (
        np.asarray(grid).reshape(-1, 3)
        if is_hwc
        else np.asarray(grid).transpose(1, 2, 0).reshape(-1, 3)
    )
    restored = np.empty_like(records)
    restored[order] = records
    if is_hwc:
        return restored.reshape(grid.shape)
    return restored.reshape(grid.shape[1], grid.shape[2], 3).transpose(2, 0, 1)


@torch.no_grad()
def _report_roundtrip_quantiles(original, reconstructed, attr_name):
    target = getattr(original, attr_name).detach().float()
    prediction = getattr(reconstructed, attr_name).detach().float()
    delta = (
        prediction.reshape(prediction.shape[0], -1)
        - target.reshape(target.shape[0], -1)
    )
    per_gaussian = delta.square().mean(1).sqrt()
    q = torch.quantile(
        per_gaussian,
        per_gaussian.new_tensor([0.50, 0.95, 0.99, 0.999]),
    )
    rmse = delta.square().mean().sqrt()
    data_range = (target.max() - target.min()).clamp_min(1e-8)
    attr_psnr = 20.0 * torch.log10(data_range / rmse.clamp_min(1e-8))
    print(
        f'[roundtrip-raw:{attr_name}] rmse={rmse.item():.7g} '
        f'psnr_range={attr_psnr.item():.3f}dB '
        f'p50={q[0].item():.7g} p95={q[1].item():.7g} '
        f'p99={q[2].item():.7g} p99.9={q[3].item():.7g} '
        f'max={per_gaussian.max().item():.7g}'
    )


@dataclass
class QuantEval:
    psnr: float
    ssim: float
    lpips: float

@dataclass
class Measurement:
    name: str
    path: str
    size_bytes: int
    quant_eval: QuantEval = None

    @property
    def human_readable_byte_size(self):
        if self.size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(np.floor(np.log(self.size_bytes) / np.log(1000)))
        p = np.power(1000, i)
        s = round(self.size_bytes / p, 2)
        return f"{s}{size_name[i]}"
    
    def to_dict(self):
        d = asdict(self)
        d.pop('quant_eval')
        if self.quant_eval is not None:
            d.update(self.quant_eval.__dict__)
        d['size'] = self.human_readable_byte_size
        return d



def log_transform(coords):
    positive = coords > 0
    negative = coords < 0
    zero = coords == 0

    transformed_coords = np.zeros_like(coords)
    transformed_coords[positive] = np.log1p(coords[positive])
    transformed_coords[negative] = -np.log1p(-coords[negative])
    # For zero, no change is needed as transformed_coords is already initialized to zeros

    return transformed_coords

def inverse_log_transform(transformed_coords):
    positive = transformed_coords > 0
    negative = transformed_coords < 0
    zero = transformed_coords == 0

    original_coords = np.zeros_like(transformed_coords)
    original_coords[positive] = np.expm1(transformed_coords[positive])
    original_coords[negative] = -np.expm1(-transformed_coords[negative])
    # For zero, no change is needed as original_coords is already initialized to zeros

    return original_coords



def get_attr_numpy(gaussians, attr_name):
    attr_tensor = gaussians.attr_as_grid_img(attr_name)
    attr_numpy = attr_tensor.detach().cpu().numpy()
    return attr_numpy


def compress_attr(attr_config, gaussians, out_folder):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    attr_params = attr_config.get('params', {})
    
    if not attr_params:
        attr_params = {}
    
    codec = codecs[attr_method]()
    attr_np = get_attr_numpy(gaussians, attr_name)
    
    file_name = f"{attr_name}.{codec.file_ending()}"
    out_file = os.path.join(out_folder, file_name)

    if attr_config.get('contract', False):
        # sc = SceneContraction()
        # TODO take the original cuda array
        # attr = torch.tensor(attr_np, device="cuda")
        # attr_contracted = sc(attr)
        # attr_np = attr_contracted.cpu().numpy()
        attr_np = log_transform(attr_np)
    
    if "quantize" in attr_config:
        quantization = attr_config["quantize"]
        min_val = attr_np.min()
        max_val = attr_np.max()
        val_range = max_val - min_val
        # no division by zero
        if val_range == 0:
            val_range = 1
        attr_np_norm = (attr_np - min_val) / (val_range)
        qpow = 2 ** quantization
        attr_np_quantized = np.round(attr_np_norm * qpow) / qpow
        attr_np = attr_np_quantized * (val_range) + min_val
        attr_np = attr_np.astype(np.float32)

    if attr_config.get('normalize', False):
        min_val, max_val = codec.encode_with_normalization(attr_np, attr_name, out_file, **attr_params)
        return file_name, min_val, max_val
    else:
        codec.encode(attr_np, out_file, **attr_params)
        return file_name, None, None


def decompress_attr(gaussians, attr_config, compressed_file, min_val, max_val):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    
    codec = codecs[attr_method]()

    if attr_config.get('normalize', False):
        decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
    else:
        decompressed_attr = codec.decode(compressed_file)

    if attr_config.get('contract', False):
        decompressed_attr = inverse_log_transform(decompressed_attr)

    # TODO dtype?
    # TODO to device?
    # TODO add grad?
    gaussians.set_attr_from_grid_img(attr_name, decompressed_attr)


def run_single_compression(gaussians, experiment_out_path, experiment_config):
    compressed_min_vals = {}
    compressed_max_vals = {}

    compressed_files = {}

    total_size_bytes = 0

    if experiment_config.get('method') == 'dual-neural-texture':
        groups = dict(experiment_config.get('neural_groups', {}) or {})
        if not groups:
            raise ValueError("dual-neural-texture requires neural_groups")
        all_neural_attributes = []
        common_params = dict(experiment_config.get('params', {}) or {})
        morton_order = getattr(gaussians, '_xyz_morton_order', None)
        permutation_file = str(
            common_params.get('xyz_morton_permutation_file', 'morton_order.npy')
        )
        if morton_order is not None:
            # Full, uncompressed uint32 index: Morton position -> original index.
            permutation_path = os.path.join(experiment_out_path, permutation_file)
            np.save(
                permutation_path,
                morton_order.detach().cpu().numpy().astype(np.uint32, copy=False),
                allow_pickle=False,
            )
            permutation_size = os.path.getsize(permutation_path)
            total_size_bytes += permutation_size
            experiment_config['xyz_morton_permutation_file'] = permutation_file
            experiment_config['xyz_morton_permutation_semantics'] = (
                'layout_position_to_original_position'
            )
            print(
                f'[xyz-layout] saved full uint32 permutation: {permutation_path} '
                f'({permutation_size / 1024**2:.3f} MiB)'
            )
        for group_name, group_config in groups.items():
            if not str(group_name).replace('_', '').replace('-', '').isalnum():
                raise ValueError(f"Unsafe neural group name: {group_name!r}")
            attributes = list(group_config.get('attributes', []) or [])
            all_neural_attributes.extend(attributes)
            neural_config = dict(common_params)
            neural_config.update(dict(group_config.get('params', {}) or {}))
            neural_config['attributes'] = attributes
            group_out = os.path.join(experiment_out_path, str(group_name))
            os.makedirs(group_out, exist_ok=True)
            print(
                f"[dual-neural:{group_name}] compressing "
                f"{[_attribute_name(item) for item in attributes]}"
            )
            group_gaussians = gaussians
            group_names = [_attribute_name(item) for item in attributes]
            if '_xyz' in group_names and morton_order is not None:
                group_gaussians = _clone_gaussians(gaussians)
                group_gaussians.prune_all_but_these_indices(morton_order)
                print('[dual-neural:xyz] using Morton layout')
            else:
                print(f'[dual-neural:{group_name}] using original SOG/PLAS layout')
            total_size_bytes += compress_gaussians(
                group_gaussians, group_out, neural_config
            )

        neural_names, _ = _validate_hybrid_attributes(all_neural_attributes, [])
        experiment_config['compressed_attributes'] = neural_names
        experiment_config['uncompressed_attributes'] = []
        experiment_config['standalone_decode'] = True
        experiment_config['max_sh_degree'] = gaussians.max_sh_degree
        experiment_config['active_sh_degree'] = gaussians.active_sh_degree
        experiment_config['disable_xyz_log_activation'] = gaussians.disable_xyz_log_activation
        with open(
            os.path.join(experiment_out_path, "compression_config.yml"),
            'w', encoding='utf-8',
        ) as stream:
            yaml.safe_dump(experiment_config, stream, sort_keys=False)
        print(
            f"[dual-neural] total deployment size: "
            f"{total_size_bytes / 1024**2:.3f} MiB"
        )
        return total_size_bytes

    if experiment_config.get('method') == 'neural-texture':
        neural_attributes = list(experiment_config.get('attributes', []))
        sog_attributes = list(experiment_config.get('sog_attributes', []))
        neural_names, sog_names = _validate_hybrid_attributes(
            neural_attributes, sog_attributes
        )

        # SH45 (or any explicitly listed neural attributes) uses the neural
        # texture/BC1 path.
        neural_config = dict(experiment_config.get('params', {}))
        neural_config['attributes'] = neural_attributes
        total_size_bytes = compress_gaussians(gaussians, experiment_out_path, neural_config)

        # The remaining attributes use the original SOG codecs and quantizers.
        for attribute in sog_attributes:
            compressed_file, min_val, max_val = compress_attr(
                attribute, gaussians, experiment_out_path
            )
            attr_name = attribute['name']
            compressed_files[attr_name] = compressed_file
            compressed_min_vals[attr_name] = min_val
            compressed_max_vals[attr_name] = max_val
            total_size_bytes += os.path.getsize(
                os.path.join(experiment_out_path, compressed_file)
            )

        pd.DataFrame(
            [compressed_min_vals, compressed_max_vals, compressed_files],
            index=["min", "max", "file"],
        ).T.to_csv(os.path.join(experiment_out_path, "compression_info.csv"))

        experiment_config['compressed_attributes'] = neural_names + sog_names
        experiment_config['uncompressed_attributes'] = []
        experiment_config['standalone_decode'] = True
        experiment_config['max_sh_degree'] = gaussians.max_sh_degree
        experiment_config['active_sh_degree'] = gaussians.active_sh_degree
        experiment_config['disable_xyz_log_activation'] = gaussians.disable_xyz_log_activation
        with open(os.path.join(experiment_out_path, "compression_config.yml"), 'w') as stream:
            yaml.safe_dump(experiment_config, stream, sort_keys=False)
        return total_size_bytes

    for attribute in experiment_config['attributes']:
        compressed_file, min_val, max_mal = compress_attr(attribute, gaussians, experiment_out_path)
        attr_name = attribute['name']
        compressed_files[attr_name] = compressed_file
        compressed_min_vals[attr_name] = min_val
        compressed_max_vals[attr_name] = max_mal
        total_size_bytes += os.path.getsize(os.path.join(experiment_out_path, compressed_file))

    compr_info = pd.DataFrame([compressed_min_vals, compressed_max_vals, compressed_files], index=["min", "max", "file"]).T
    compr_info.to_csv(os.path.join(experiment_out_path, "compression_info.csv"))

    experiment_config['max_sh_degree'] = gaussians.max_sh_degree
    experiment_config['active_sh_degree'] = gaussians.active_sh_degree
    experiment_config['disable_xyz_log_activation'] = gaussians.disable_xyz_log_activation
    with open(os.path.join(experiment_out_path, "compression_config.yml"), 'w') as stream:
        yaml.dump(experiment_config, stream)

    return total_size_bytes

def run_compressions(gaussians, out_path, compr_exp_config):

    # TODO some code duplciation with run_experiments / run_roundtrip

    results = {}

    for experiment in compr_exp_config['experiments']:

        experiment_name = experiment['name']
        experiment_out_path = os.path.join(out_path, experiment_name)
        os.makedirs(experiment_out_path, exist_ok=True)

        size_bytes = run_single_compression(gaussians, experiment_out_path, experiment)
        results[f"size_bytes/cmpr_{experiment['name']}"] = size_bytes

    return results

def run_single_decompression(compressed_dir, template_gaussians=None):
    with open(
        os.path.join(compressed_dir, "compression_config.yml"),
        'r',
        encoding='utf-8',
    ) as stream:
        experiment_config = yaml.safe_load(stream)

    decompressed_gaussians = GaussianModel(experiment_config['max_sh_degree'], experiment_config['disable_xyz_log_activation'])
    decompressed_gaussians.active_sh_degree = experiment_config['active_sh_degree']

    if experiment_config.get('method') == 'dual-neural-texture':
        groups = dict(experiment_config.get('neural_groups', {}) or {})
        decoded_names = []
        common_params = dict(experiment_config.get('params', {}) or {})
        permutation_name = experiment_config.get('xyz_morton_permutation_file')
        layout_to_original = None
        if permutation_name:
            permutation_path = os.path.join(compressed_dir, permutation_name)
            if not os.path.isfile(permutation_path):
                raise FileNotFoundError(
                    f'Missing XYZ Morton permutation: {permutation_path}'
                )
            permutation_np = np.load(permutation_path, allow_pickle=False)
            if permutation_np.dtype != np.uint32 or permutation_np.ndim != 1:
                raise ValueError(
                    'XYZ Morton permutation must be a one-dimensional uint32 array'
                )
            layout_to_original = torch.from_numpy(
                permutation_np.astype(np.int64, copy=False)
            )
            print(
                f'[xyz-layout] loaded full permutation: {permutation_path} '
                f'({layout_to_original.numel()} entries)'
            )
        for group_name, group_config in groups.items():
            attributes = list(group_config.get('attributes', []) or [])
            neural_config = dict(common_params)
            neural_config.update(dict(group_config.get('params', {}) or {}))
            neural_config['attributes'] = attributes
            group_dir = os.path.join(compressed_dir, str(group_name))
            print(f"[dual-neural:{group_name}] decoding {group_dir}")
            decoded = decompress_gaussians(group_dir, neural_config)
            for attr_name, decoded_attr in decoded.items():
                if attr_name in decoded_names:
                    raise RuntimeError(f"Attribute decoded twice: {attr_name}")
                if attr_name == '_xyz' and layout_to_original is not None:
                    decoded_attr = _restore_grid_from_layout(
                        decoded_attr,
                        layout_to_original,
                    )
                    print('[xyz-layout] restored decoded XYZ to original SOG/PLAS order')
                decompressed_gaussians.set_attr_from_grid_img(attr_name, decoded_attr)
                decoded_names.append(attr_name)
        missing = sorted(set(GAUSSIAN_ATTRIBUTE_NAMES) - set(decoded_names))
        if missing:
            raise RuntimeError(f"Dual neural stream is incomplete; missing {missing}")
        return decompressed_gaussians

    if experiment_config.get('method') == 'neural-texture':
        sog_attributes = list(experiment_config.get('sog_attributes', []))
        if sog_attributes:
            compr_info = pd.read_csv(
                os.path.join(compressed_dir, "compression_info.csv"), index_col=0
            )
            for attribute in sog_attributes:
                attr_name = attribute["name"]
                compressed_file = os.path.join(
                    compressed_dir, compr_info.loc[attr_name, "file"]
                )
                decompress_attr(
                    decompressed_gaussians,
                    attribute,
                    compressed_file,
                    compr_info.loc[attr_name, "min"],
                    compr_info.loc[attr_name, "max"],
                )
        else:
            # Backward compatibility for old SH-only experiment directories.
            # Such streams are not standalone and still need the source model.
            _copy_gaussian_template(decompressed_gaussians, template_gaussians)

        neural_config = dict(experiment_config.get('params', {}))
        neural_config['attributes'] = experiment_config.get('attributes', [])
        for attr_name, decoded_attr in decompress_gaussians(compressed_dir, neural_config).items():
            decompressed_gaussians.set_attr_from_grid_img(attr_name, decoded_attr)
        return decompressed_gaussians

    compr_info = pd.read_csv(os.path.join(compressed_dir, "compression_info.csv"), index_col=0)

    for attribute in experiment_config['attributes']:
        attr_name = attribute["name"]
        # compressed_bytes = compressed_attrs[attr_name]
        compressed_file = os.path.join(compressed_dir, compr_info.loc[attr_name, "file"])

        decompress_attr(decompressed_gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"])

    return decompressed_gaussians

def run_decompressions(compressions_dir, template_gaussians=None):
    
    for compressed_dir in os.listdir(compressions_dir):
        compressed_dir_path = os.path.join(compressions_dir, compressed_dir)
        if not os.path.isdir(compressed_dir_path):
            continue
        yield os.path.basename(compressed_dir_path), run_single_decompression(
            compressed_dir_path, template_gaussians=template_gaussians
        )

def run_roundtrip(gaussians, out_path, experiment_config):

    experiment_name = experiment_config['name']
    experiment_out_path = os.path.join(out_path, experiment_name)
    os.makedirs(experiment_out_path, exist_ok=True)

    gaussians.prune_to_square_shape()
    morton_order = _prepare_morton_block_layout(gaussians, experiment_config)
    gaussians._xyz_morton_order = morton_order
    
    total_size_bytes = run_single_compression(gaussians, experiment_out_path, experiment_config)
    
    decompressed_gaussians = run_single_decompression(
        experiment_out_path, template_gaussians=gaussians
    )

    for attr_name in GAUSSIAN_ATTRIBUTE_NAMES:
        _report_roundtrip_quantiles(gaussians, decompressed_gaussians, attr_name)

    return decompressed_gaussians, total_size_bytes, experiment_out_path






def run_experiments(training_cfg, cmdline_iteration, compr_exp_config, disable_lpips=False):

    gaussians = GaussianModel(training_cfg.dataset.sh_degree, False)

    scene = Scene(training_cfg.dataset, gaussians, load_iteration=cmdline_iteration, shuffle=False)
    iteration = scene.loaded_iter

    gaussians._xyz = gaussians.inverse_xyz_activation(gaussians._xyz.detach())

    print(f"Compressing {training_cfg.dataset.model_path} iteration {iteration}")
    out_path = os.path.join(training_cfg.dataset.model_path, "compression", f"iteration_{iteration}")
    os.makedirs(out_path, exist_ok=True)

    bg_color = [1,1,1] if training_cfg.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    all_cameras = scene.getTestCameras() # + scene.getTrainCameras()

    def render_test_measure(gaussians_to_render):

        with torch.inference_mode():
            psnrs = []
            ssims = []
            lpipss = []

            for idx, view in enumerate(all_cameras):
                rendering = render(view, gaussians_to_render, training_cfg.pipeline, background)["render"]
                gt = view.original_image[0:3, :, :]
                psnrs.append(psnr(rendering, gt).cpu().numpy())
                ssims.append(ssim(rendering, gt).cpu().numpy())
                if disable_lpips:
                    lpipss.append(np.nan)
                else:
                    lpipss.append(lpips(rendering, gt, net_type='vgg').cpu().numpy())
        
        return QuantEval(psnr=np.mean(psnrs), ssim=np.mean(ssims), lpips=np.mean(lpipss))

    exp_results = []

    original_eval = render_test_measure(gaussians)
    exp_results.append(Measurement(name="PLY", path=scene.loaded_gaussian_ply, size_bytes=os.path.getsize(scene.loaded_gaussian_ply), quant_eval=original_eval))

    for experiment in compr_exp_config['experiments']:
        gaussians_roundtrip, compressed_size_bytes, exp_out_path = run_roundtrip(gaussians, out_path, experiment)
        rendered_eval = render_test_measure(gaussians_roundtrip)
        meas = Measurement(name=experiment['name'], path=exp_out_path, size_bytes=compressed_size_bytes, quant_eval=rendered_eval)
        print(meas)
        exp_results.append(meas)

    exp_df = pd.DataFrame([m.to_dict() for m in exp_results])

    sorted_columns_for_easy_comparison = ['name', 'size', 'psnr', 'ssim', 'lpips', 'path', 'size_bytes']

    assert len(exp_df.columns) == len(sorted_columns_for_easy_comparison), "Hey, you added a column to the dataframe, please add it to the sorted_columns_for_easy_comparison list as well"

    exp_df = exp_df[sorted_columns_for_easy_comparison]
    exp_df.to_csv(os.path.join(out_path, "results.csv"), index=False)
    return exp_df
        



def load_config(config_path: str):
    # Config files are UTF-8 (the YAML contains Chinese comments), while
    # Windows Anaconda Prompt commonly defaults to the GBK code page.
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    return config


def compression_exp():
    # example args: --model_path=../models/truck --iteration 10000 --compression_config compression/configs/jpeg_xl.yml [--results_csv results.csv] [--disable_lpips]

    parser = ArgumentParser(description="Compression script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--compression_config", type=str)
    parser.add_argument("--results_csv", type=str)
    parser.add_argument("--results_tex", type=str)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument(
        "--experiments",
        type=str,
        default="",
        help="Comma-separated experiment names from the compression YAML. "
             "Empty runs every experiment.",
    )
    
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)

    iteration = args_cmdline.iteration
    model_path = args_cmdline.model_path

    compr_exp_config = load_config(args_cmdline.compression_config)
    if args_cmdline.experiments:
        requested = {
            name.strip() for name in args_cmdline.experiments.split(",")
            if name.strip()
        }
        available = {
            experiment["name"] for experiment in compr_exp_config["experiments"]
        }
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                f"Unknown --experiments values {missing}; available={sorted(available)}"
            )
        compr_exp_config["experiments"] = [
            experiment for experiment in compr_exp_config["experiments"]
            if experiment["name"] in requested
        ]

    training_cfg = get_hydra_training_args(model_path)

    training_cfg.dataset.model_path = model_path
    training_cfg.dataset.source_path = args_cmdline.source_path

    disable_lpips = args_cmdline.disable_lpips

    results_csv = args_cmdline.results_csv
    results_tex = args_cmdline.results_tex

    exp_df = run_experiments(training_cfg, iteration, compr_exp_config, disable_lpips=disable_lpips)
    print(exp_df)

    if results_csv:
        csv_dirname = os.path.dirname(results_csv)
        if csv_dirname:
            os.makedirs(csv_dirname, exist_ok=True)
        exp_df.to_csv(results_csv, index=False)

    if results_tex:
        tex_dirname = os.path.dirname(results_tex)
        if tex_dirname:
            os.makedirs(tex_dirname, exist_ok=True)
        exp_df.to_latex(results_tex, index=False,
                        columns=["name", "psnr", "ssim", "lpips", "size_bytes"],
                        header=["Name", "PSNR $\\uparrow$", "SSIM $\\uparrow$", "LPIPS $\\downarrow$", "Size (MB)"],
                        formatters={"size_bytes": lambda x: f"{x / 1000 / 1000:.2f}", "psnr": lambda x: f"{x:.2f}", "ssim": lambda x: f"{x:.3f}", "lpips": lambda x: f"{x:.3f}"}
                        )

    
    

if __name__ == "__main__":
    compression_exp()
