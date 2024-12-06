#!/bin/bash
MODEL_NAME="$1"

python convert_data.py \
    --model $MODEL_NAME \
    --data-type fp16 \
    --driver kuae \
    --driver-version kuae_1.2.0 \
    --backend musa \
    --backend-version 1.2.0 \
    --engine mtt \
    --serving vllm \
    --gpu 'MTT S4000' \
    --base-dir result_outputs
