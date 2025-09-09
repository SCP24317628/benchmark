MODEL_PATH="/data/models/deepseek-ai/deepseek-v3.1"
# 输入输出对
IO_PAIRS=(
    "16384 16384"
    "8192 8192"
)
# 并发数和 num-prompts 一一对应
CONCURRENCY_AND_PROMPTS=(
    "1 4"
    "2 4"
    "4 8"	
    "8 16"
    "16 32"
    "32 64"
    "50 100"
    "64 128"
    "100 200"
    "128 256"
)
DATASET_NAME="random"

MODEL_NAME=$(basename "${MODEL_PATH%/}" | tr '[:upper:]' '[:lower:]')
OUTPUT_FILE="vllm_bench_${MODEL_NAME}_results.csv"
RESULT_DIR="/workspace/vllm_results"
BASE_DIR=output_result
FULL_OUTPUT_PATH="$BASE_DIR/$OUTPUT_FILE"
echo "" > $FULL_OUTPUT_PATH  # 清空文件

# 表头
HEADER="input_len,output_len,max_concurrency,num_prompts,Successful_requests,Benchmark_duration_s,Total_input_tokens,Total_generated_tokens,Request_throughput_req_s,Output_token_throughput_tok_s,Total_Token_throughput_tok_s,Mean_TTFT_ms,Median_TTFT_ms,P99_TTFT_ms,Mean_TPOT_ms,Median_TPOT_ms,P99_TPOT_ms,Mean_ITL_ms,Median_ITL_ms,P99_ITL_ms,Mean_E2EL_ms,Median_E2EL_ms,P99_E2EL_ms"
echo $HEADER >> $FULL_OUTPUT_PATH

# 循环 IO_PAIRS
for PAIR in "${IO_PAIRS[@]}"; do
    read W O <<< "$PAIR"

    # 循环并发和对应的 num-prompts
    for CP in "${CONCURRENCY_AND_PROMPTS[@]}"; do
        read C N <<< "$CP"
        echo "===== Running benchmark: input_len=$W, output_len=$O, max_concurrency=$C, num_prompts=$N ====="

        RESULT=$(vllm bench serve \
            --model $MODEL_PATH \
            --served-model-name $MODEL_NAME \
            --dataset-name $DATASET_NAME \
            --random-input-len $W \
            --random-output-len $O \
            --num-prompts $N \
            --ignore-eos \
            --save-result \
            --percentile-metrics 'ttft,tpot,itl,e2el' \
            --metric-percentiles "95,99" \
            --result-dir $RESULT_DIR \
            --max-concurrency $C \
            2>&1 | awk '/============ Serving Benchmark Result ============/{flag=1; next} /==================================================/{flag=0} flag')

        CSV_LINE=$(echo "$RESULT" | awk '
            function clean(s){gsub(/^[ \t]+|[ \t]+$/,"",s); return s}
            /Successful requests:/ {sr=clean($NF)}
            /Benchmark duration/ {bd=clean($NF)}
            /Total input tokens/ {ti=clean($NF)}
            /Total generated tokens/ {tg=clean($NF)}
            /Request throughput/ {rt=clean($NF)}
            /Output token throughput/ {ot=clean($NF)}
            /Total Token throughput/ {tt=clean($NF)}
            /Mean TTFT/ {mttft=clean($NF)}
            /Median TTFT/ {medttft=clean($NF)}
            /P99 TTFT/ {p99ttft=clean($NF)}
            /Mean TPOT/ {mtpot=clean($NF)}
            /Median TPOT/ {medtpot=clean($NF)}
            /P99 TPOT/ {p99tpot=clean($NF)}
            /Mean ITL/ {mitl=clean($NF)}
            /Median ITL/ {meditl=clean($NF)}
            /P99 ITL/ {p99itl=clean($NF)}
            /Mean E2EL/ {me2el=clean($NF)}
            /Median E2EL/ {mede2el=clean($NF)}
            /P99 E2EL/ {p99e2el=clean($NF)}
            END {
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                "'$W'","'$O'","'$C'","'$N'",
                sr, bd, ti, tg, rt, ot, tt,
                mttft, medttft, p99ttft,
                mtpot, medtpot, p99tpot,
                mitl, meditl, p99itl,
                me2el, mede2el, p99e2el
            }'
        )

        echo "$CSV_LINE" >> $FULL_OUTPUT_PATH

        MEAN_E2EL=$(echo $CSV_LINE | cut -d',' -f21)
        echo ">>> Completed: input_len=$W, output_len=$O, max_concurrency=$C, num_prompts=$N, Mean_E2EL_ms=$MEAN_E2EL"
        echo
    done
done

echo "All benchmarks finished. Results saved to $FULL_OUTPUT_PATH."

