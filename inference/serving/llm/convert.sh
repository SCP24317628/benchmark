#!/bin/bash
MODEL_NAME="$1"
MODEL_ALIAS="$2"

python convert_data.py \
    --model $MODEL_NAME \
    --model-alias $MODEL_ALIAS \
    --data-type FP16 \
    --driver 'musa_driver' \
    --driver-version '2.7.0' \
    --backend musa \
    --backend-version rc3.1.0 \
    --engine mtt \
    --engine-version 0.2.1 \
    --serving vllm \
    --serving-version 0.4.2 \
    --gpu 'S4000_0x0327' \
    --gpu-num 8 \
    --tp 8 \
    --base-dir result_outputs \
    --source 'in-house_benchmark'