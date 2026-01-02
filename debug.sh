#!/bin/bash

python patchtst_distill.py \
  --data_path datasets/ETTh1.csv \
  --context_len 96 --horizon 24 \
  --cache_stride 24 \
  --debug_samples 3 \
  --epochs 1 --batch_size 32 --device auto