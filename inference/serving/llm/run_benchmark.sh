#!/bin/bash
MODEL_NAME="$1"
DEVICE="$2"


python benchmark_llm.py \
    --model $MODEL_NAME \
    --device $DEVICE \
    --config musa/config.yaml \
    --input 64,128,256,512 \
    --output 64,128,192,256 \
    --concurrent 1,4,16,32,64,128