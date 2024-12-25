#!/bin/bash
MODEL_NAME="$1"
MODEL_ALIAS="$2"

python convert_data.py \
    --model $MODEL_NAME \
    --model-alias $MODEL_ALIAS \
    --data-type fp16 \
    --driver kuae \
    --driver-version release-kuae-1.2.0 \
    --backend musa \
    --backend-version 2.1.0 \
    --engine mtt \
    --engine-version 0.14.0 \
    --serving vllm \
    --serving-version 0.4.2 \
    --gpu 'MTT S4000' \
    --base-dir result_outputs