#!/bin/bash
MODEL_NAME="$1"
MODEL_ALIAS="$2"
python3 convert_vLLM_bench.py \
    --model $MODEL_NAME \
    --model-alias $MODEL_ALIAS \
    --tp 8 \
    --dp 1 \
    --pp 1 \
    --ep 1 \
    --data-type fp16 \
    --gpu 'H20' \
    --gpu-num 8 \
    --driver 'NVIDIA-Linux-x86' \
    --driver-version '570.124.06' \
    --backend cuda \
    --backend-version 12.8 \
    --engine cuda \
    --engine-version 12.8 \
    --serving vllm \
    --serving-version 0.7.3 \
    --source 'vllm'