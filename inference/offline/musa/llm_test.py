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

import mttransformer
import torch
import torch_musa
import math
import argparse
import time
import pandas as pd
import os
from enum import IntEnum
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Any, Union
from itertools import accumulate

@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch.
    max_seq_len: Optional[int]
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    # INFO(ljjia): max_blocks_per_seq = ceil_div(max_seq_len in model config/tokens_per_page)
    block_tables: Optional[torch.Tensor]

@dataclass
class FlashAttentionMetadata(PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch.
    max_query_len: Optional[int]
    # Maximum sequence length in the batch.
    max_seq_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

@dataclass
class AttentionMetadata:
    """Attention metadata for prefill and decode batched together."""
    # Total number of prefill requests.
    num_prefills: int
    # Number of prefill tokens.
    num_prefill_tokens: int
    # Number of decode tokens. Note that it is equivalent to the number of
    # decode requests.
    num_decode_tokens: int
    # The attention metadata for prefill requests in a batch.
    # None if there's no prefill requests in a batch.
    prefill_metadata: FlashAttentionMetadata
    # The attention metadata for decode requests in a batch.
    # None if there's no decode requests in a batch.
    decode_metadata: FlashAttentionMetadata
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor
    # The kv cache's data type.
    kv_cache_dtype: str

    def __post_init__(self):
        if self.num_prefill_tokens > 0:
            assert self.num_prefills > 0
            assert self.prefill_metadata is not None
        if self.num_decode_tokens > 0:
            assert self.decode_metadata is not None


def _struct_fa_metadata(input_ids: List, history_tokens: List = None):
    query_len = list(map(len, input_ids))
    query_num = len(query_len)
    seq_lens = query_len
    if history_tokens != None:
        assert len(query_len) == len(history_tokens)
        seq_lens = list(map(sum, zip(query_len, history_tokens)))

    seq_lens_tensor = torch.tensor(seq_lens)
    max_query_len = max(query_len) if query_len else 0
    bPrefill = max_query_len > 1
    max_seq_len = max(seq_lens) if seq_lens else 0
    subquery_start_loc = torch.tensor(list(accumulate(query_len, initial=0)), dtype=torch.int32)
    seq_start_loc = torch.tensor(list(accumulate(seq_lens, initial=0)), dtype=torch.int32)
    context_lens_tensor = (
        torch.zeros(query_num, dtype=torch.int32)
        if history_tokens == None
        else torch.tensor(history_tokens)
    )
    fa_metadata = FlashAttentionMetadata(
        # PagedAttentionMetadata
        seq_lens_tensor = seq_lens_tensor,
        max_seq_len = max_seq_len,
        block_tables = None, # page table set None first, chg in outside
        # FlashAttentionMetadata
        is_prompt = bPrefill,
        seq_lens = seq_lens,
        max_query_len = max_query_len,
        subquery_start_loc = subquery_start_loc,
        seq_start_loc = seq_start_loc,
        context_lens_tensor = context_lens_tensor,
        use_cuda_graph = False,
    )
    return fa_metadata

def struct_pagedattn_metadata(prefill_ids, decode_ids, decode_his, tokens_per_page = 64):
  if prefill_ids != None:
    prefill_metadata = _struct_fa_metadata(prefill_ids)
    num_prefills = len(prefill_ids)
    num_prefill_tokens = sum(prefill_metadata.seq_lens)
    return AttentionMetadata(
      num_prefills = num_prefills,
      num_prefill_tokens = num_prefill_tokens,
      num_decode_tokens = 0,
      prefill_metadata = prefill_metadata,
      decode_metadata = None,
      slot_mapping = None, # set None first, chg in outside
      kv_cache_dtype = 'fp16',
    )
  else:
    assert(decode_ids != None and decode_his != None)
    decode_metadata = _struct_fa_metadata(decode_ids, decode_his)
    num_decode_tokens = len(decode_ids)
    return AttentionMetadata(
      num_prefills = 0,
      num_prefill_tokens = 0,
      num_decode_tokens = num_decode_tokens,
      prefill_metadata = None,
      decode_metadata = decode_metadata,
      slot_mapping = None, # set None first, chg in outside
      kv_cache_dtype = 'fp16',
    )



def run_test(cmodel, model_instant, batch, num_prefill_token, num_decode_token):
  tp = cmodel.model.tensor_para_size
  tokens_per_page = cmodel.model.tokens_per_page

  page_num = math.ceil((num_prefill_token + num_decode_token) / tokens_per_page)
  need_pages = page_num * batch
  kv_cache = [torch.zeros(
    need_pages,
    2,
    cmodel.model.layer_num,
    cmodel.model.num_key_value_heads // tp,
    tokens_per_page,
    cmodel.model.size_per_head_pad,
    dtype = torch.float16,
    device = f'musa:{i}')
    for i in range(tp)]

  prefill_ids = [[1,] * prefill_token,] * batch
  block_tables = torch.zeros((batch, 0), dtype=torch.int32)
  slot_mapping = [b * page_num * tokens_per_page + t for b in range(batch) for t in range(num_prefill_token)]
  attn_meta = struct_pagedattn_metadata(prefill_ids, None, None, tokens_per_page)
  attn_meta.prefill_metadata.block_tables = block_tables
  attn_meta.slot_mapping = torch.tensor(slot_mapping, dtype = torch.int64)

  prefill_ids = torch.tensor([e for row in prefill_ids for e in row], dtype = torch.int64)
  position_ids = [i for i in range(num_prefill_token)] * batch
  position_ids = torch.tensor(position_ids, dtype = torch.int32)

  for i in range(1): # warm up
    output = model_instant.forward(prefill_ids, position_ids, kv_cache, attn_meta)
    sample_out = model_instant.post_process(output, temperature = 0.0, top_p = 0.9).cpu()
    if num_decode_token > 0:
      decode_ids = [[1]] * batch
      pid = num_prefill_token + 0
      decode_his = [pid] * batch
      slot_mapping = [b * page_num * tokens_per_page + pid for b in range(batch)]
      block_tables = [[b * page_num + p for p in range(math.ceil(pid / tokens_per_page))] for b in range(batch)]
      decode_attn_meta = struct_pagedattn_metadata(None, decode_ids, decode_his, tokens_per_page)
      decode_attn_meta.decode_metadata.block_tables = torch.tensor(block_tables, dtype = torch.int32)
      decode_attn_meta.slot_mapping = torch.tensor(slot_mapping, dtype = torch.int64)
      decode_ids = torch.tensor([e for row in decode_ids for e in row], dtype = torch.int64)
      decode_position_ids = [pid] * batch
      decode_position_ids = torch.tensor(decode_position_ids, dtype = torch.int32)
      output = model_instant.forward(decode_ids, decode_position_ids, kv_cache, decode_attn_meta)
      sample_out = model_instant.post_process(output, temperature = 0.0, top_p = 0.9).cpu()
  # torch.musa.synchronize()

  prefill_start = time.time()
  count = 1
  for i in range(count): # prefill
    output = model_instant.forward(prefill_ids, position_ids, kv_cache, attn_meta)
    sample_out = model_instant.post_process(output, temperature = 0.0, top_p = 0.9).cpu()
  prefill_duration = (time.time() - prefill_start) / count

  decode_time = 0.0
  for i in range(1): # decode
    for d in range(num_decode_token):
      decode_ids = [[1]] * batch
      pid = num_prefill_token + d
      decode_his = [pid] * batch
      slot_mapping = [b * page_num * tokens_per_page + pid for b in range(batch)]
      block_tables = [[b * page_num + p for p in range(math.ceil(pid / tokens_per_page))] for b in range(batch)]
      decode_attn_meta = struct_pagedattn_metadata(None, decode_ids, decode_his, tokens_per_page)
      decode_attn_meta.decode_metadata.block_tables = torch.tensor(block_tables, dtype = torch.int32)
      decode_attn_meta.slot_mapping = torch.tensor(slot_mapping, dtype = torch.int64)

      decode_ids = torch.tensor([e for row in decode_ids for e in row], dtype = torch.int64)
      decode_position_ids = [pid] * batch
      decode_position_ids = torch.tensor(decode_position_ids, dtype = torch.int32)

      decode_start = time.time()
      output = model_instant.forward(decode_ids, decode_position_ids, kv_cache, decode_attn_meta)
      sample_out = model_instant.post_process(output, temperature = 0.0, top_p = 0.9).cpu()
      decode_duration = time.time() - decode_start
      decode_time += decode_duration
  decode_duration = decode_time / 1
  return prefill_duration, decode_duration

def int2list(val):
    if isinstance(val, int):
        return [val]
    else:
        return val

if __name__ == "__main__":
    df = pd.DataFrame(
        columns=[
            "Model",
            "Data_Type",
            "GPU_Num",
            "num_heads", # head_num
            "num_kv_heads", # num_key_value_heads
            "num_layers", # layer_num
            "tensor_para_size", # tensor_para_size,
            "pipeline_para_size", # pipeline_para_size
            "vocab_size", # vocab_size
            "quantization", # QuantMode.0
            "batch",
            "prefill_tokens",
            "decode_tokens",
            "prefill_latency(ms)",
            "generated_tokens", # total decode tokens
            "generation_time(ms)", # decode latency
            "latency(ms)", # end 2 end latency
            "tokens_per_second", # end 2 end tps
            "single_batch_decode_tps",
            "total_decode_tps",
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size-list", type=int, nargs="+", default=8, help="batch size list"
    )
    parser.add_argument(
        "--prefill-len-list",
        type=int,
        nargs="+",
        default=512,
        help="prefill sequence length list",
    )
    parser.add_argument(
        "--decode-len-list", type=int, nargs="+", default=64, help="decode length"
    )
    parser.add_argument(
        "--model-name", type=str, default="default model", help="model name"
    )
    perf_args, unknow_args = parser.parse_known_args()
    prefill_token_list = int2list(perf_args.prefill_len_list)
    decode_token_list = int2list(perf_args.decode_len_list)
    batch_list = int2list(perf_args.batch_size_list)
    model_name = perf_args.model_name
    print(f"batch {batch_list} | prefill_token {prefill_token_list} | decode_token {decode_token_list}")

    args = mttransformer.get_args(unknow_args)
    cmodel = mttransformer.LLMEngine(args)
    model_instant = cmodel.create_model_instant()
    dtype = "FP16"
    gpu_num = model_instant.tensor_para_size

    # prefill_token = [[1,]*8]*1
    for batch in batch_list:
      for prefill_token in prefill_token_list:
        for decode_token in decode_token_list:
          time.sleep(0.5 if prefill_token * batch < 1024 * 8 else 1)
          print(f"batch: {batch}, prefill_token: {prefill_token}, decode_token: {decode_token} ")
          # prefill output one token, so need decrease 1
          true_decode_token = decode_token - 1
          prefill_duration, decode_duration = run_test(cmodel, model_instant, batch, prefill_token, true_decode_token)
          tps = true_decode_token / decode_duration if true_decode_token else 0
          batch_tps = tps * batch
          
          # model configs
          head_num = cmodel.model.head_num
          num_key_value_heads = cmodel.model.num_key_value_heads if cmodel.model.num_key_value_heads else "None"
          layer_num = cmodel.model.layer_num
          tensor_para_size = cmodel.model.tensor_para_size
          pipeline_para_size = cmodel.model.pipeline_para_size
          vocab_size = cmodel.model.vocab_size
          
          # add new metrics, align with trt-llm results
          generated_tokens = true_decode_token * batch
          generation_time = decode_duration
          latency = prefill_duration + decode_duration
          tokens_per_second = generated_tokens / latency 
          quantization = 'QuantMode.' + cmodel.model.quant_mode if cmodel.model.quant_mode else "QuantMode.0"

          df.loc[len(df.index)] = [
              model_name,
              dtype,
              gpu_num,
              head_num,
              num_key_value_heads,
              layer_num,
              tensor_para_size,
              pipeline_para_size,
              vocab_size,
              quantization, # QuantMode.0, QuantMode.1 (int8), QuantMode.2 (int4)
              batch,
              prefill_token,
              decode_token,
              f"{prefill_duration * 1000:.2f}",
              f"{generated_tokens}", # total decode tokens
              f"{generation_time * 1000:.2f}", # decode latency
              f"{latency * 1000:.2f}", # end 2 end latency
              f"{tokens_per_second:.2f}", # end 2 end tps
              f"{tps:.2f}",
              f"{batch_tps:.2f}", # decode tps
          ]
    os.system("mkdir -p perf_data")
    df.to_csv(f"perf_data/{model_name}.csv", index=False)
    print(df)
    print(f"{model_name} perf finish")
    print("=========")