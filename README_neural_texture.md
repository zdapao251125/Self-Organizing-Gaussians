# 替换文件

将文件夹内的文件复制到 `Self-Organizing-Gaussians` 目录下，使用以下命令运行


启动代码

  python train.py \
  --config-name ours_q_sh_local_test \
  'hydra.run.dir=outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}-${run.name}' \
  dataset.source_path=<DATASET_PATH>/truck \
  'run.compress_iterations=[30000]' \
  run.no_progress_bar=false \
  run.name=truck-neural-texture


压缩部分独立启动代码（适用于已经有训练完的ply数据）

  python -m compression.compression_exp \
  --model_path <MODEL_PATH> \
  --source_path <DATASET_PATH>/truck \
  --iteration 30000 \
  --compression_config config/compression/neural_texture.yaml \
  --disable_lpips

debug需要查看详细信息，修改config/compression中的neural_texture.yaml,将下列属性改为：

  params:
    verbose_debug: true
    debug_every: 2000
    debug_outlier_topk: 10
    save_debug_reference: true