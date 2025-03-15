#!/bin/bash
MODEL_NAME="$1"
DEVICE="$2"


python benchmark_llm.py \
    --model $MODEL_NAME \
    --device $DEVICE \
    --config $DEVICE/config.yaml \
    --proxy true \
    --dry true \
    --input 128,256,512,1024 \
    --output 128,256,512,1024 \
    --concurrent 1,4,8,16,32,64,128