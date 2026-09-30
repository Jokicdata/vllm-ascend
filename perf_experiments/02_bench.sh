#!/bin/bash
# vLLM 压测脚本 (客户端打点, 测 TTFT/TPOT/ITL)
#
# 用法:
#   bash 02_bench.sh                                  # 默认: 1024 in / 128 out / 64 请求 / 速率 8
#   TAG=cp_4096 bash 02_bench.sh 4096 64 16 2         # 长 prompt 负载
#
# 参数顺序: [input_len] [output_len] [num_prompts] [request_rate]
#
# 说明:
#   TAG        实验标签, 决定结果文件名, 默认 baseline
#   结果落在 /workspace/perf_test/results/, 后续用 03_summary.sh 汇总对比

TAG=${TAG:-baseline}
IN=${1:-1024}
OUT=${2:-128}
NUM=${3:-64}
RATE=${4:-8}
RESULT_DIR=/workspace/perf_test/results
mkdir -p $RESULT_DIR

python -m vllm.benchmarks.serve \
  --backend openai-chat \
  --model Qwen3-32B \
  --host 127.0.0.1 --port 8998 \
  --dataset-name random \
  --random-input-len $IN --random-output-len $OUT \
  --num-prompts $NUM --request-rate $RATE \
  --percentile-metrics ttft,tpot,itl \
  --save-result \
  --result-dir $RESULT_DIR \
  --result-file ${TAG}_in${IN}_out${OUT}_n${NUM}_r${RATE}.json

echo "result saved to $RESULT_DIR/${TAG}_in${IN}_out${OUT}_n${NUM}_r${RATE}.json"
