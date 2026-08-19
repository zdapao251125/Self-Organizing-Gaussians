# Neural Texture Baseline for SOG

This branch replaces SOG's final per-attribute JPEG/JPEG XL compression
experiment with the baseline from `D:\xym_neural_text`:

- four learned BC1 latent texture pyramids;
- a small per-scene MLP decoder;
- quantization-aware BC1 training with straight-through gradients;
- reconstruction L1 only, with no CDF, entropy, likelihood, bitrate, or rate
  loss.

The 3DGS optimization, pruning, square-grid conversion, PLAS sorting, and
rendering code are unchanged. At each configured compression iteration, the
sorted Gaussian attribute grids are normalized and concatenated, then the
neural baseline is optimized offline.

## Compressed representation

The measured runtime representation contains only:

- `latent_00.bc1.dds` through `latent_03.bc1.dds`;
- `decoder_fp16.bin`;
- `metadata.json`;
- `gaussian_layout.json`.

SOG's round-trip evaluation decodes these files directly. It does not load a
PyTorch checkpoint. Optional checkpoints, previews, and training history are
excluded from the compressed byte count.

The default latent layout is BCF1 VarA, derived from the current Gaussian grid:
`[W, W, W/2, W/2]`, rounded to BC1's 4x4 alignment.

## Full run

Run from an Anaconda Prompt:

```bat
cd /d D:\SOG_neural_baseline
conda activate sogs
set "WANDB_MODE=online"
python train.py --config-name ours_q_sh_local_test "dataset.source_path=D:\SOG\tandt\truck" "hydra.run.dir=D:\SOG_neural_baseline_output\truck" "run.name=truck_neural_texture_baseline"
```

The default configuration trains SOG for 30,000 iterations. Neural compression
runs at iterations 7,000, 10,000, 20,000, and 30,000, with 100,000 BC1/MLP
iterations plus 1,000 quantized decoder-finetuning iterations each time. This
is computationally expensive. For a final comparison, normally compress only
at iteration 30,000:

```bat
python train.py --config-name ours_q_sh_local_test "dataset.source_path=D:\SOG\tandt\truck" "hydra.run.dir=D:\SOG_neural_baseline_output\truck_final" "run.name=truck_neural_texture_baseline_final" "run.compress_iterations=[30000]"
```

## Scope

This is an offline compression baseline. It does not implement the HPG 2025
paper's DirectX/Vulkan Cooperative Vectors, tile classification, G-buffer
inference, or shader-time random access. SOG reconstructs the complete Gaussian
attribute arrays before rendering.
