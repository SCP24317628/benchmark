# Copyright @2020-2023 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.
#
# This software ("this software and its documentations" or "the software") is
# protected by Copyright and the information contained herein is confidential.
#
# The software contained herein is PROPRIETARY to Moore Threads and is being
# provided under the terms and conditions of a form of Moore Threads software
# license agreement by and between Moore Threads and Licensee ("License
# Agreement") or electronically accepted by Licensee. Notwithstanding any
# terms or conditions to the contrary in the License Agreement, copy or
# disclosure of the software to any third party without the express written
# consent of Moore Threads is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
# AGREEMENT, MOORE THREDS MAKES NO REPRESENTATION ABOUT ANY WARRANTIES,
# INCLUDING BUT NOT LIMITED TO THE SUITABILITY OF THE SOFTWARE FOR ANY
# PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF
# ANY KIND. MOORE THREDS DISCLAIMS ALL WARRANTIES WITH REGARD TO THE
# SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL MOORE THREDS BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THE SOFTWARE.

#!/usr/bin/env python3
import os, sys
import pandas as pd
import json
import argparse
import time


def gen_num_list(li):
    ret = ""
    for x in li:
        ret += f"{x} "
    return ret


def run_perf(config):
    for model in config:
        print(model)
        max_seq_len = (
            max(model["prefill_token_lens"]) + max(model["decode_token_lens"])
        )
        max_batch_size = max(model["batchs"])
        max_tokens = max_batch_size * max_seq_len
        batch_size_list = gen_num_list(model["batchs"])
        prefill_token_lens = gen_num_list(model["prefill_token_lens"])
        decode_token_lens = gen_num_list(model["decode_token_lens"])
        mtt_cmd = f"python -u llm_test.py --model-name={model['model_name']} --checkpoint-path='{model['path']}' --batch-size-list {batch_size_list} --prefill-len-list {prefill_token_lens} --decode-len-list {decode_token_lens} --max-seq-len {max_seq_len} --max-batch-size={max_batch_size} --max-tokens {max_tokens}"
        cmd = f"MUSA_BLOCK_SCHEDULE_MODE=1 {mtt_cmd}"
        print(cmd)
        os.system(cmd)
        time.sleep(10)


def combine_ret(config, output_path):
    dfs = []
    for model in config:
        name = model["model_name"]
        csv_path = f"perf_data/{name}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.loc[len(df.index)] = ["" for _ in range(len(df.columns))]
            dfs.append(df)
    ret = pd.concat(dfs)
    print(ret)
    ret.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mtt perf script')
    parser.add_argument('config_path', type=str, default="perf_config.json", help="input config json file path")
    parser.add_argument('-o', '--output-path', type=str, default="perf_data/combine_ret.csv", help="output file path")
    args = parser.parse_args()
    # os.system("rm -r perf_data")
    with open(args.config_path) as f:
        config = json.load(f)
        run_perf(config)
        combine_ret(config, args.output_path)