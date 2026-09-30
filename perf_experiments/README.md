# vLLM-Ascend 性能实验脚本集（8 卡 910B4）

配套教程见 [docs/ai_infra_experiments_tutorial.md](../docs/ai_infra_experiments_tutorial.md)。

## 文件清单

| 文件 | 用途 | 对应实验 |
|---|---|---|
| `01_start_server.sh` | 启动 vllm serve（支持 TAG / EXTRA_ARGS 注入） | 全部 |
| `02_bench.sh` | 压测（客户端打点 TTFT/TPOT/ITL，结果存 json） | 全部 |
| `03_summary.sh` | 汇总 results/*.json 为一张对比表 | 全部 |
| `plot_probe.py` | 画 batch 组成探针图（Sarathi Figure-1 自制复现） | 实验 1 |

## 使用前准备

在服务器上把本目录复制到 `/workspace/perf_test/`（或直接 clone 仓库）：

```bash
mkdir -p /workspace/perf_test
cp perf_experiments/*.sh perf_experiments/*.py /workspace/perf_test/
cd /workspace/perf_test
```

确认模型路径：脚本里写的是 `/workspace/models/Qwen3-32B`，如有出入先改 `01_start_server.sh`。

## 语法速查：TAG=xxx 是什么

```bash
TAG=cp_4096 bash 01_start_server.sh
```

`TAG=cp_4096` 是 bash 的**临时环境变量前缀**：只在本条命令执行期间生效，脚本内部用 `${TAG:-baseline}` 读取（未设置时取默认值）。它的作用是给实验起名——日志、结果文件都会带上这个标签，多个实验互不覆盖。`EXTRA_ARGS` 同理，用于把额外 vllm 参数注入启动命令，实现"一份脚本、一次只改一个变量"。

## 实验 0：基线

```bash
bash 01_start_server.sh
TAG=baseline bash 02_bench.sh 1024 128 64 8
TAG=baseline_long bash 02_bench.sh 4096 64 16 1
bash 03_summary.sh
```

## 实验 1：Chunked Prefill

```bash
# A组: 关闭
EXTRA_ARGS="--no-enable-chunked-prefill" TAG=cp_off bash 01_start_server.sh
TAG=cp_off bash 02_bench.sh 4096 64 16 2
TAG=cp_off_short bash 02_bench.sh 128 256 64 8

# B/C组: 开启, 预算 4096 / 8192
EXTRA_ARGS="--enable-chunked-prefill --max-num-batched-tokens 4096" TAG=cp_4096 bash 01_start_server.sh
TAG=cp_4096 bash 02_bench.sh 4096 64 16 2
TAG=cp_4096_short bash 02_bench.sh 128 256 64 8

EXTRA_ARGS="--enable-chunked-prefill --max-num-batched-tokens 8192" TAG=cp_8192 bash 01_start_server.sh
TAG=cp_8192 bash 02_bench.sh 4096 64 16 2
TAG=cp_8192_short bash 02_bench.sh 128 256 64 8
```

探针（需先按教程给 `vllm_ascend/worker/model_runner_v1.py` 的 `execute_model` 加探针代码并 `pip install -e`）：

```bash
MY_PROBE=1 EXTRA_ARGS="--enable-chunked-prefill --max-num-batched-tokens 4096" TAG=probe bash 01_start_server.sh
TAG=probe bash 02_bench.sh 4096 64 16 2
python3 plot_probe.py /workspace/perf_test/probe_batch.csv
```

注意：本实验不要同时开 `--enable-prefix-caching`（910B 上历史上存在兼容性限制）。

## 实验 2：Prefix Caching

```bash
EXTRA_ARGS="" TAG=apc_off bash 01_start_server.sh
TAG=apc_off bash 02_bench.sh 2048 64 32 2

EXTRA_ARGS="--enable-prefix-caching --block-size 128" TAG=apc_on bash 01_start_server.sh
TAG=apc_on bash 02_bench.sh 2048 64 32 2
```

注意：`--dataset-name random` 前缀命中率为 0，测不出收益；验证收益需用共享前缀负载（教程里有固定 system prompt 的 curl 循环示例）。

## 实验 3：ACLGraph 图模式

```bash
# B组: 去掉 enforce-eager 开图模式（基线 A 组已有）
EXTRA_ARGS="" TAG=graph_on bash 01_start_server.sh
TAG=graph_on bash 02_bench.sh 128 256 64 8

# 精简 capture sizes（需先按教程加 [EXP-ACLGRAPH] 日志行）
EXTRA_ARGS="--compilation-config '{\"cudagraph_capture_sizes\": [1,2,4,8]}'" TAG=g_small bash 01_start_server.sh
TAG=g_small bash 02_bench.sh 128 256 64 8
grep "EXP-ACLGRAPH" /workspace/perf_test/serve_g_small.log
```

对比三件事：启动耗时（日志时间戳）、TPOT（压测 json）、显存峰值（`npu-smi info`）。

## 实验 4：投机解码

```bash
EXTRA_ARGS="--speculative-config '{\"method\":\"ngram\",\"prompt_lookup_num\":4,\"num_speculative_tokens\":3}'" TAG=spec_ngram bash 01_start_server.sh
TAG=spec_ngram bash 02_bench.sh 128 256 32 4
grep -i "accept" /workspace/perf_test/serve_spec_ngram.log
```

接受率 > 50% 时 TPOT 应下降；< 30% 说明收益为负。ngram 在 NPU 后端不一定完整支持，报错请查当前版本投机解码支持矩阵。

## 结果汇总

```bash
bash 03_summary.sh                                  # 全部
bash 03_summary.sh "results/cp_*"                   # 只看实验 1
```
