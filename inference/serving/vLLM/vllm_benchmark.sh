#!/bin/bash
# Model configuration
# Tokens length configuration
INPUT_LIST=(128 256 512 1024) # can extension more
OUTPUT_LIST=(128) # can extension more
CONCURRENCY_LIST=(1 4 8 16 32 64 128) # Concurrency settings, can extension more
NUM_PROMPTS=256 # Test num prompts


DATASET_NAME="random"

MODEL_PATH="$1"
# Check if the model path exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model path does not exist - ${MODEL_PATH}"
    exit 1
fi
# Dynamically generate the model name (in lowercase)
MODEL_NAME=$(basename "${MODEL_PATH%/}" | tr '[:upper:]' '[:lower:]')

OUTPUT_DIR="output_result"
mkdir -p "${OUTPUT_DIR}"
# Modify the output file path
OUTPUT_FILE="${OUTPUT_DIR}/vllm_bench_${MODEL_NAME}_results.csv"
echo "" > "${OUTPUT_FILE}"  # Empty the file

# CSV header
HEADER="input_len,output_len,max_concurrency,num_prompts,Successful_requests,Benchmark_duration_s,Total_input_tokens,Total_generated_tokens,Request_throughput_req_s,Output_token_throughput_tok_s,Total_Token_throughput_tok_s,Mean_TTFT_ms,Median_TTFT_ms,P99_TTFT_ms,Mean_TPOT_ms,Median_TPOT_ms,P99_TPOT_ms,Mean_ITL_ms,Median_ITL_ms,P99_ITL_ms,Mean_E2EL_ms,Median_E2EL_ms,P99_E2EL_ms"
echo $HEADER >> $OUTPUT_FILE

# loops to generate all combinations
for W in "${INPUT_LIST[@]}"; do
    for O in "${OUTPUT_LIST[@]}"; do
        for C in "${CONCURRENCY_LIST[@]}"; do
                echo "===== Running benchmark: input_len=$W, output_len=$O, max_concurrency=$C, num_prompts=$NUM_PROMPTS ====="
                # Execute benchmark and capture results
                RESULT=$(vllm bench serve \
                    --model $MODEL_PATH \
                    --served-model-name $MODEL_NAME \
                    --dataset-name $DATASET_NAME \
                    --random-input-len $W \
                    --random-output-len $O \
                    --num-prompts $NUM_PROMPTS \
                    --ignore-eos \
                    --save-result \
                    --percentile-metrics 'ttft,tpot,itl,e2el' \
                    --result-dir $RESULT_DIR \
                    --max-concurrency $C \
                    2>&1 | awk '/============ Serving Benchmark Result ============/{flag=1; next} /==================================================/{flag=0} flag')

                # Extract numbers values and generate CSV row
                CSV_LINE=$(echo "$RESULT" | awk -F':' '
                    function clean(s){gsub(/^[ \t]+|[ \t]+$/,"",s); return s}
                    /Successful requests/ {sr=clean($2)}
                    /Benchmark duration/ {bd=clean($2)}
                    /Total input tokens/ {ti=clean($2)}
                    /Total generated tokens/ {tg=clean($2)}
                    /Request throughput/ {rt=clean($2)}
                    /Output token throughput/ {ot=clean($2)}
                    /Total Token throughput/ {tt=clean($2)}
                    /Mean TTFT/ {mttft=clean($2)}
                    /Median TTFT/ {medttft=clean($2)}
                    /P99 TTFT/ {p99ttft=clean($2)}
                    /Mean TPOT/ {mtpot=clean($2)}
                    /Median TPOT/ {medtpot=clean($2)}
                    /P99 TPOT/ {p99tpot=clean($2)}
                    /Mean ITL/ {mitl=clean($2)}
                    /Median ITL/ {meditl=clean($2)}
                    /P99 ITL/ {p99itl=clean($2)}
                    /Mean E2EL/ {me2el=clean($2)}
                    /Median E2EL/ {mede2el=clean($2)}
                    /P99 E2EL/ {p99e2el=clean($2)}
                    END {
                        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                        "'$W'","'$O'","'$C'","'$NUM_PROMPTS'",
                        sr, bd, ti, tg, rt, ot, tt,
                        mttft, medttft, p99ttft,
                        mtpot, medtpot, p99tpot,
                        mitl, meditl, p99itl,
                        me2el, mede2el, p99e2el
                    }')

                # write in CSV
                echo "$CSV_LINE" >> $OUTPUT_FILE

                # print log, align columns
                MEAN_E2EL=$(echo $CSV_LINE | cut -d',' -f21)
                echo ">>> Completed: input_len=$W, output_len=$O, max_concurrency=$C, num_prompts=$NUM_PROMPTS, Mean_E2EL_ms=$MEAN_E2EL"
            done
        done
    done
done

echo "All benchmarks finished. Results saved to ${OUTPUT_FILE}"

