#!/bin/bash
MODEL_NAME="$1"
SAVE_DIR="$2"
test -n "$MODEL_NAME"
export OPENAI_API_KEY=openai
export OPENAI_API_BASE="http://0.0.0.0:4000/v1"
export HF_ENDPOINT=https://hf-mirror.com

python llmperf/token_benchmark_ray.py \
--model $MODEL_NAME \
--mean-input-tokens 2048 \
--stddev-input-tokens 24 \
--mean-output-tokens 128 \
--stddev-output-tokens 10 \
--max-num-completed-requests 100 \
--timeout 600 \
--num-concurrent-requests 10 \
--results-dir "result_outputs/$SAVE_DIR" \
--llm-api openai