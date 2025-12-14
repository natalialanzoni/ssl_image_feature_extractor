#!/bin/bash

set -x
python run_model.py \
  --reuse_unzipped\
  --resume ./checkpoints_resnet_birds_adam_v2.convnext/resnet_moco_latest.pt \
  --temperature 0.07\
  --color_jitter .6\
  --unzipped_dir /cluster/synth/bruce/dataset_cache/unzipped \
  --zip_dir /cluster/synth/bruce/dataset_cache/zips \
  --lr 0.0001 \
  --warmup_epochs 20 \
  --opt adam \
  --name ADAM_CONVNEXT_RES\
  --convnext\
  --output_dir ./checkpoints_resnet_birds_adam_v2.convnext_res\
  --epochs 10000 \
  --no_scale_lr_by_batch

#  --hf_cache_dir /cluster/synth/bruce/huggingface \
