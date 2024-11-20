#!/bin/bash
MODEL_NAME="$1"
test -n "$MODEL_NAME"
MODEL_DIR="/data/mtt/model_convert/$MODEL_NAME"
test -d "$MODEL_DIR"
export HF_ENDPOINT=https://hf-mirror.com

python -O -u -m vllm.entrypoints.openai.api_server \
    --host=127.0.0.1 \
    --port=8000 \
    --model=/data/mtt/model_convert/$MODEL_NAME \
    --max-model-len=4096 \
    --tokenizer=hf-internal-testing/llama-tokenizer \
    --api-key=openai \
    --trust-remote-code \
    --tensor-parallel-size=1 \
    --block-size=64 \
    -pp=1 \
    --gpu-memory-utilization=0.95 \
    --device="musa"