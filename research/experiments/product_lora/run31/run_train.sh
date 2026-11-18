#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python train.py \
  --pretrained_model_name_or_path=model --variant=fp16 \
  --instance_data_dir=data/train --output_dir=outputs/train \
  --instance_prompt='a photo of sksdrill red and black cordless power drill' \
  --resolution=768 --center_crop --train_batch_size=1 \
  --max_train_steps=200 --rank=4 --gradient_checkpointing \
  --mixed_precision=fp16 --learning_rate=0.0001 \
  --lr_scheduler=constant --lr_warmup_steps=0 --seed=31 \
  --checkpointing_steps=1000 --dataloader_num_workers=0
