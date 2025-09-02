import subprocess
import csv
import itertools
import os

MODEL_PATH = "/data/models/deepseek-ai/deepseek-r1-distill-qwen-1.5b"
MODEL_NAME = "deepseek-r1-distill-qwen-1.5b"
RESULT_DIR = "/workspace/vllm_results"
DATASET_NAME = "random"

INPUT_LIST = [128, 256, 512, 1024]
OUTPUT_LIST = [128]
CONCURRENCY_LIST = [1, 4, 8, 16, 32, 64, 128]
NUM_PROMPTS = 256

# Linux环境下创建output_result目录
OUTPUT_DIR = "output_result"
# 使用Linux兼容的方式确保目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置Linux格式的输出文件路径
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"vllm_bench_{MODEL_NAME}_results.csv")

HEADER = [
    "input_len","output_len","max_concurrency","num_prompts",
    "Successful_requests","Benchmark_duration_s","Total_input_tokens","Total_generated_tokens",
    "Request_throughput_req_s","Output_token_throughput_tok_s","Total_Token_throughput_tok_s",
    "Mean_TTFT_ms","Median_TTFT_ms","P99_TTFT_ms",
    "Mean_TPOT_ms","Median_TPOT_ms","P99_TPOT_ms",
    "Mean_ITL_ms","Median_ITL_ms","P99_ITL_ms",
    "Mean_E2EL_ms","Median_E2EL_ms","P99_E2EL_ms"
]


def run_benchmark(W, O, C, N):
    print(f"===== Running benchmark: input_len={W}, output_len={O}, max_concurrency={C}, num_prompts={N} =====")

    cmd = [
        "vllm", "bench", "serve",
        "--model", MODEL_PATH,
        "--served-model-name", "deepseek-r1-distill-llama-70b",
        "--dataset-name", DATASET_NAME,
        "--random-input-len", str(W),
        "--random-output-len", str(O),
        "--num-prompts", str(N),
        "--ignore-eos",
        "--save-result",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--result-dir", RESULT_DIR,
        "--max-concurrency", str(C)
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = proc.stdout.splitlines()

    # 抓取 Serving Benchmark Result 部分 - 改进版
    capture = []
    flag = False
    print("=== 原始输出开始 ===")
    for line in lines:
        print(line)  # 打印原始输出，方便调试
        if "============ Serving Benchmark Result ============" in line:
            flag = True
            continue
        if "==================================================" in line and flag:
            break
        if flag and line.strip():
            capture.append(line.strip())
    print("=== 原始输出结束 ===")
    
    print(f"=== 捕获的结果行: {len(capture)} 行 ===")

    data = {h: "" for h in HEADER}
    # 改进的数据映射，完全匹配日志中的字段名称
    data_map = {
        "Successful requests": "Successful_requests",
        "Benchmark duration (s)": "Benchmark_duration_s",  # 注意这里加上了 (s)
        "Total input tokens": "Total_input_tokens",
        "Total generated tokens": "Total_generated_tokens",
        "Request throughput (req/s)": "Request_throughput_req_s",  # 注意这里加上了 (req/s)
        "Output token throughput (tok/s)": "Output_token_throughput_tok_s",  # 注意这里加上了 (tok/s)
        "Total Token throughput (tok/s)": "Total_Token_throughput_tok_s",  # 注意这里加上了 (tok/s)
        "Mean TTFT (ms)": "Mean_TTFT_ms",  # 注意这里加上了 (ms)
        "Median TTFT (ms)": "Median_TTFT_ms",  # 注意这里加上了 (ms)
        "P99 TTFT (ms)": "P99_TTFT_ms",  # 注意这里加上了 (ms)
        "Mean TPOT (ms)": "Mean_TPOT_ms",  # 注意这里加上了 (ms)
        "Median TPOT (ms)": "Median_TPOT_ms",  # 注意这里加上了 (ms)
        "P99 TPOT (ms)": "P99_TPOT_ms",  # 注意这里加上了 (ms)
        "Mean ITL (ms)": "Mean_ITL_ms",  # 注意这里加上了 (ms)
        "Median ITL (ms)": "Median_ITL_ms",  # 注意这里加上了 (ms)
        "P99 ITL (ms)": "P99_ITL_ms",  # 注意这里加上了 (ms)
        "Mean E2EL (ms)": "Mean_E2EL_ms",  # 注意这里加上了 (ms)
        "Median E2EL (ms)": "Median_E2EL_ms",  # 注意这里加上了 (ms)
        "P99 E2EL (ms)": "P99_E2EL_ms",  # 注意这里加上了 (ms)
    }
    
    # 改进的解析逻辑
    for line in capture:
        print(f"解析行: {line}")  # 打印正在解析的行
        if ":" not in line:
            continue
        
        # 分割键值对，只分割第一个冒号
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
            
        key = parts[0].strip()
        val = parts[1].strip()
        
        # 查找匹配的键
        matched = False
        for log_key, csv_key in data_map.items():
            if key == log_key:
                data[csv_key] = val
                print(f"匹配到: {log_key} -> {csv_key} = {val}")
                matched = True
                break
        
        if not matched:
            print(f"未匹配的键: {key}")

    # 固定参数
    data["input_len"] = W
    data["output_len"] = O
    data["max_concurrency"] = C
    data["num_prompts"] = N

    print(f">>> Completed: input_len={W}, output_len={O}, max_concurrency={C}, num_prompts={N}, Mean_E2EL_ms={data['Mean_E2EL_ms']}")
    
    # 打印所有提取的数据用于调试
    print("=== 提取的数据摘要 ===")
    for key, value in data.items():
        if value:  # 只打印有值的字段
            print(f"{key}: {value}")
    print()

    return data


def main():
    # 显示Linux格式的输出文件路径
    print(f"输出文件将保存到: {os.path.abspath(OUTPUT_FILE)}")
    
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()

        for W, O, C, N in itertools.product(INPUT_LIST, OUTPUT_LIST, CONCURRENCY_LIST, [NUM_PROMPTS]):
            row = run_benchmark(W, O, C, N)
            writer.writerow(row)

    print(f"All benchmarks finished. Results saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()