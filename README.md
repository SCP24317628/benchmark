# benchmark
Test suites for serving &amp; offline benchmark
## Benchmark structure for llm
### Inference
 - serving
    ##### work with serving task
    cd inference/serving/llm
    // suppose you are trying to benchmark glm-4-9b with configs:
        1. 'inp: 64; out: 64; concurrent: 4'
        2. 'inp: 64; out: 64; concurrent: 8'
        3. 'inp: 128; out: 64; concurrent: 4'
        4. 'inp: 128; out: 64; concurrent: 8'
    // code
    ```
    python benchmark_llm.py --config config.yaml --model glm-4-9b --device musa --input "64,128" --output "64" --concurrent "4,8"
    ```
 - offline

