#!/bin/bash
# 不需要部署服务就加 --dry
MODEL_NAME="$1"
DEVICE="$2"


python benchmark_onlyvllm.py \
    --model $MODEL_NAME \
    --device $DEVICE \
    --dry  \  
    --input 128,256,1024,2048,4096 \
    --output 128 \
    --concurrent 256,512,1024
