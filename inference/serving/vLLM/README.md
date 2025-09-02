# vLLM Serve 部署及vllm_benchmark.sh脚本使用教程

# 下面将介绍如何使用 vLLM 进行模型部署的基本步骤和命令示例。

vllm serve /data/models/deepseek-ai/deepseek-r1-distill-qwen-1.5b     --host 0.0.0.0     --port 8000      --tokenizer /data/models/deepseek-ai/deepseek-r1-distill-qwen-1.5b     --api-key openai     --tensor-parallel-size 1     --gpu-memory-utilization 0.95     --dtype bfloat16     --served-model-name deepseek-r1-distill-qwen-1.5b  --device cuda
参数说明 ：


   模型路径 ：首个位置参数指定模型存储路径

   --host ：服务监听地址（0.0.0.0表示开放所有网络接口）

   --port ：服务监听端口

   --tokenizer ：分词器路径（通常与模型路径一致）

   --api-key ：API访问密钥（生产环境建议使用复杂密钥）

   --tensor-parallel-size ：张量并行度（需匹配GPU数量）

   --gpu-memory-utilization ：显存利用率（0.95表示保留5%显存余量）

#以vllm --version查看版本0.7.3为例
vllm serve \
  [MODEL_PATH] \
  --host [HOST_IP] \
  --port [PORT] \
  --tokenizer [TOKENIZER_PATH] \
  --api-key [API_KEY] \
  --trust-remote-code \
  --tensor-parallel-size [GPU_NUM] \
  --gpu-memory-utilization [MEM_UTIL] \
  --dtype [PRECISION] \
  --served-model-name [MODEL_NAME] \
  --device [DEVICE]

# vllm_benchmark.sh脚本使用教程
修改需要的测试的
INPUT_LIST=(128 256 512 1024) 
OUTPUT_LIST=(128) 
CONCURRENCY_LIST=(1 4 8 16 32 64 128) 
NUM_PROMPTS= 256

