#!/bin/bash
# vLLM-Ascend 实验服务启动脚本 (8卡 910B4, Qwen3-32B)
#
# 用法:
#   bash 01_start_server.sh                          # 基线启动 (TAG=baseline)
#   TAG=cp_4096 EXTRA_ARGS="--enable-chunked-prefill --max-num-batched-tokens 4096" bash 01_start_server.sh
#
# 说明:
#   TAG          实验标签, 决定日志文件名 (serve_${TAG}.log), 默认 baseline
#   EXTRA_ARGS   注入到 vllm serve 的额外参数, 用于控制变量实验
#
# 语法说明: "TAG=xxx bash 01_start_server.sh" 是 bash 的临时环境变量前缀,
# 只在本条命令生效, 脚本内部用 ${TAG:-baseline} 读取 (不存在时取默认值)。

TAG=${TAG:-baseline}
LOG=/workspace/perf_test/serve_${TAG}.log

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ACL_OP_INIT_MODE=1

# 杀掉旧服务实例, 保证每次实验从干净状态启动
pkill -f "vllm serve" 2>/dev/null
sleep 5

vllm serve /workspace/models/Qwen3-32B \
  --host 0.0.0.0 --port 8998 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enforce-eager \
  $EXTRA_ARGS > $LOG 2>&1 &

echo "waiting for server... (log: $LOG)"
for i in $(seq 1 180); do
  curl -s http://127.0.0.1:8998/v1/models > /dev/null && { echo "READY"; exit 0; }
  sleep 10
done
echo "TIMEOUT - check $LOG"
exit 1
