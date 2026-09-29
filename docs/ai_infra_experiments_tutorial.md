# AI Infra 论文实践教程：在 8 卡 910B4 上优化 vLLM-Ascend（面向小白）

本文是"从经典论文到代码实践"的入门教程，目标读者：刚学完 AI Infra 理论、想在真实硬件（8 卡 Ascend 910B4）上动手验证的新人。

每个实验都遵循同一个闭环：

```
读论文思想 → 拨参数/改代码 → 压测对比 → 得出量化结论
```

核心心法（来自 AI Infra 学习指南的"不可能三角"）：

> 每个优化都是"牺牲 A 换 B"。没有基线数据的优化都是玄学；一次实验只改一个变量。

## 实验总览

| 实验 | 对应论文 | 优化原理一句话 | 主要收益指标 | 动手程度 |
|---|---|---|---|---|
| 0 基线 | — | 建立测量口径 | — | 只跑命令 |
| 1 Chunked Prefill | [Sarathi-Serve (OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/agrawal) | 长 prompt 切片，与 decode 混批 | ITL P99 / TTFT P99 | 参数 + 探针代码 |
| 2 Prefix Caching | [vLLM/PagedAttention (SOSP'23)](https://arxiv.org/abs/2309.06180)、[SGLang (NeurIPS'24)](https://arxiv.org/abs/2312.07104) | 复用相同前缀的 KV cache | TTFT | 参数 |
| 3 ACLGraph 图模式 | CUDA Graphs 思想 | 图录制回放消除 CPU 调度开销 | TPOT / ITL | 参数 + 1 行代码 |
| 4 投机解码 | [Speculative Sampling (ICML'23)](https://arxiv.org/abs/2302.01318) | 小模型起草 + 大模型并行验证 | TPOT | 参数 |

涉及源码路径（vllm-ascend 仓库内，已核实）：

- `vllm_ascend/worker/model_runner_v1.py` — 每步计算的执行入口（`execute_model`）
- `vllm_ascend/attention/utils.py` — chunked prefill workspace 分配策略
- `vllm_ascend/compilation/acl_graph.py` — ACLGraph 图捕获与回放
- `vllm_ascend/ascend_config.py` — 图模式 capture sizes 配置读取
- `vllm_ascend/envs.py` — 所有 `VLLM_ASCEND_*` 环境变量定义

---

## 实验 0：基线（一切优化的分母）

### 为什么先跑基线

TTFT（首 token 延迟）由多段构成：排队 + prefill 计算 + 首包回传。不做客户端打点、没有基线对比，后面所有实验都无法判断"是变好了还是变坏了"。

### 部署步骤

在服务器上创建实验目录：

```bash
mkdir -p /workspace/perf_test/results && cd /workspace/perf_test
```

创建 `01_start_server.sh`（服务启动脚本）：

```bash
#!/bin/bash
# 用法: EXTRA_ARGS="--xxx" TAG=cp4096 bash 01_start_server.sh
TAG=${TAG:-baseline}
LOG=/workspace/perf_test/serve_${TAG}.log

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ACL_OP_INIT_MODE=1

pkill -f "vllm serve" 2>/dev/null; sleep 5

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

echo "waiting for server..."
for i in $(seq 1 180); do
  curl -s http://127.0.0.1:8998/v1/models > /dev/null && { echo "READY"; exit 0; }
  sleep 10
done
echo "TIMEOUT - check $LOG"; exit 1
```

创建 `02_bench.sh`（压测脚本，客户端打点）：

```bash
#!/bin/bash
# 用法: TAG=baseline bash 02_bench.sh [input_len] [output_len] [num] [rate]
TAG=${TAG:-baseline}
IN=${1:-1024}; OUT=${2:-128}; NUM=${3:-64}; RATE=${4:-8}

python -m vllm.benchmarks.serve \
  --backend openai-chat \
  --model Qwen3-32B \
  --host 127.0.0.1 --port 8998 \
  --dataset-name random \
  --random-input-len $IN --random-output-len $OUT \
  --num-prompts $NUM --request-rate $RATE \
  --percentile-metrics ttft,tpot,itl \
  --save-result \
  --result-dir /workspace/perf_test/results \
  --result-file ${TAG}_in${IN}_out${OUT}_n${NUM}_r${RATE}.json
```

### 执行

```bash
bash 01_start_server.sh                        # 基线启动
TAG=baseline bash 02_bench.sh 1024 128 64 8    # 标准负载
TAG=baseline_long bash 02_bench.sh 4096 64 16 1  # 长 prompt 负载
```

### 产出

`results/baseline_*.json`，记录 TTFT / TPOT / ITL 的 P50 / P99，作为后续所有实验的对照组。

---

## 实验 1：Chunked Prefill（Sarathi-Serve, OSDI'24）

- 论文：https://www.usenix.org/conference/osdi24/presentation/agrawal
- 关联源码：`vllm_ascend/attention/utils.py` 中的 `ascend_chunked_prefill_workspace_size()`

### 背景与原理

LLM 推理分两个阶段：

- **Prefill**：一次性处理整个 prompt，矩阵大、算力利用率高（compute-bound），决定了 TTFT
- **Decode**：逐 token 生成，每步矩阵退化为向量，瓶颈在显存带宽（memory-bound），决定了 ITL/TPOT

问题在于：一个 8K 长 prompt 的 prefill 要"一口气"算完，期间后面所有请求只能干等。

```
关闭 chunked prefill (默认长 prefill 独占 GPU):
时间 ──────────────────────────────────────────────►
请求A(8K长prompt): [======prefill 独占3秒======][decode...]
请求B(短prompt):        ↑↑↑ 到了但只能干等 3 秒 ↑↑↑
请求C(短prompt):            ↑↑↑ 也干等 ↑↑↑
                        └── B/C 的 TTFT 全爆炸 ──┘

开启 chunked prefill (切成 2048 一片, 和 decode 混批):
时间 ──────────────────────────────────────────────►
A的切片:  [=片1=][=片2=][=片3=][=片4=][decode...]
B:          [prefill+decode 同批跑]  TTFT 很快出首字
C:             [同批]
          └── 每一"步"被切成固定 token 预算 ──┘
              A 自己的 TTFT 变慢(被切片), 但全系统 P99 大幅改善
```

本质是 Scheduler 每步只处理 `max_num_batched_tokens` 个 token（预结算力预算），长 prefill 不再"霸占"一步计算，decode 请求每一两步就能被调度一次，ITL 不再有秒级尖刺。

**牺牲了什么换来什么**：牺牲单个长请求的 TTFT，换取全体请求的尾延迟（P99）和 ITL 平滑。

### 第 1 步：参数实验（不改代码）

```bash
# A组: 关闭 chunked prefill
EXTRA_ARGS="--no-enable-chunked-prefill" TAG=cp_off bash 01_start_server.sh
TAG=cp_off bash 02_bench.sh 4096 64 16 2
TAG=cp_off_short bash 02_bench.sh 128 256 64 8

# B组: 开启, token 预算 4096（910B 社区经验: 256 的倍数, 4096 起步）
EXTRA_ARGS="--enable-chunked-prefill --max-num-batched-tokens 4096" TAG=cp_4096 bash 01_start_server.sh
TAG=cp_4096 bash 02_bench.sh 4096 64 16 2
TAG=cp_4096_short bash 02_bench.sh 128 256 64 8

# C组: 预算 8192
EXTRA_ARGS="--enable-chunked-prefill --max-num-batched-tokens 8192" TAG=cp_8192 bash 01_start_server.sh
TAG=cp_8192 bash 02_bench.sh 4096 64 16 2
TAG=cp_8192_short bash 02_bench.sh 128 256 64 8
```

注意：910B 上 chunked prefill 与 prefix caching 历史上存在兼容性限制，本实验不要同时开 `--enable-prefix-caching`。

### 第 2 步：加探针，亲眼看"混批"（第一次代码修改）

```bash
cd /workspace/vllm-ascend        # clone + pip install -e 的源码目录
git checkout -b exp1-chunked-probe
```

定位执行入口：

```bash
grep -n "def execute_model" vllm_ascend/worker/model_runner_v1.py
# 输出: 2216:    def execute_model(
```

在 `execute_model` 方法体第一行插入探针：

```python
def execute_model(self, scheduler_output):
    # ===== MY-PROBE: 记录每步 batch 组成 =====
    import os, time
    if os.environ.get("MY_PROBE") == "1":
        try:
            _n_reqs = len(scheduler_output.scheduled_requests)
            with open("/workspace/perf_test/probe_batch.csv", "a") as f:
                f.write(f"{time.time():.3f},{_n_reqs}\n")
        except AttributeError:
            print("[PROBE] fields:", [a for a in dir(scheduler_output) if not a.startswith("_")])
    # ===== MY-PROBE END =====
    ...  # 原有代码
```

字段名随版本可能不同：先直接 `print(dir(scheduler_output))` 确认有哪些属性再填写。

```bash
git add -A && git commit -m "exp1: add batch composition probe"

MY_PROBE=1 EXTRA_ARGS="--enable-chunked-prefill --max-num-batched-tokens 4096" TAG=probe bash 01_start_server.sh
TAG=probe bash 02_bench.sh 4096 64 16 2
```

对 CSV 画图：关闭 chunked 时会看到少数几步 token 数巨大（长 prefill 独占）；开启后每步被削平到预算附近。这就是 Sarathi 论文 Figure 1 的自制复现。

### 验证标准

| 指标 | 预期 |
|---|---|
| 长 prompt 混合负载的 ITL P99 | 显著下降（decode 不再被长 prefill 卡住） |
| 长 prompt 自身的 TTFT | 略升（切片代价） |
| 短 prompt 高并发 TTFT P99 | 下降 |
| 吞吐 | 持平或略升 |

---

## 实验 2：Prefix Caching（PagedAttention + RadixAttention）

- 论文：vLLM https://arxiv.org/abs/2309.06180 ｜ SGLang https://arxiv.org/abs/2312.07104
- 关联源码：前缀命中逻辑在 vLLM 主框架 `vllm/v1/core/kv_cache_manager.py`；vllm-ascend 侧关注 block size 对齐（`vllm_ascend/attention/`）

### 背景与原理

多轮对话/固定系统提示词场景，请求的前缀部分完全一样，KV cache 算过一次就可以复用。

先理解 KV block：PagedAttention 借鉴操作系统虚拟内存"分页"思想，把 KV cache 切成固定大小的 block（910B 推荐 128 token/块），对每个 block 计算哈希。新请求的前缀逐块匹配哈希，命中即复用——这就是 vLLM 中 "v"（virtual memory，虚拟内存）的来源。

```
多轮对话场景, 请求都带同一段长系统提示词(S):

无 prefix cache:
  请求1: [S(2000tok) + 问题1]  → S 的 KV 全部重算   TTFT=800ms
  请求2: [S(2000tok) + 问题2]  → S 的 KV 又重算一遍  TTFT=800ms  ← 重复劳动!

有 prefix cache:
  请求1: [S + 问题1]  → S 的 KV 算完, 按 block 存进哈希表
  请求2: [S + 问题2]  → 逐 block 算哈希 → 命中! 直接复用
                        只需 prefill 问题2的几十个 token
                        TTFT=80ms  ← 降约10倍
                        └─ 前提: S 的 KV block 还没被淘汰
```

**牺牲了什么换来什么**：牺牲少量哈希查找开销和显存管理复杂度，换来重复前缀场景的 TTFT 大幅下降。

### 执行

```bash
# A组: 关闭（默认）
EXTRA_ARGS="" TAG=apc_off bash 01_start_server.sh
TAG=apc_off bash 02_bench.sh 2048 64 32 2

# B组: 开启 + 910B 推荐 block size
EXTRA_ARGS="--enable-prefix-caching --block-size 128" TAG=apc_on bash 01_start_server.sh
TAG=apc_on bash 02_bench.sh 2048 64 32 2
```

注意压测负载：`--dataset-name random` 的随机 prompt 前缀命中率为 0，测不出收益。要测收益需用共享前缀的负载，最简单的办法是用固定 system prompt 发 chat 请求：

```bash
# 循环发 100 个带相同 system prompt 的请求
for i in $(seq 1 100); do
  curl -s http://127.0.0.1:8998/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "Qwen3-32B",
    "messages": [
      {"role": "system", "content": "你是一个资深AI Infra工程师...(写一段约2000 token的长系统提示)"},
      {"role": "user", "content": "问题'"$i"': 请简述chunked prefill的原理"}
    ],
    "max_tokens": 32
  }' -o /dev/null -w "req$i ttft=%{time_starttransfer}s\n"
done > /workspace/perf_test/apc_curl_test.txt
```

对比 `apc_off` / `apc_on` 两次的 `time_starttransfer` 分布（这就是客户端口径的 TTFT）。

### 进阶小实验

`--block-size 16` vs `--block-size 128` 各跑一组——910B 上 128 对齐 FlashAttention 算子更友好，实测验证。

### 验证标准

| 指标 | 预期 |
|---|---|
| 共享前缀请求的 TTFT | 与命中前缀长度成比例下降（2000 token 前缀命中时可达数倍改善） |
| 不共享前缀的请求 | 持平（少量哈希计算开销） |

---

## 实验 3：ACLGraph 图模式（第一个真正的代码改动实验）

- 论文思想来源：CUDA Graphs（NVIDIA），vllm-ascend 的等价实现是 ACLGraph
- 关联源码：
  - `vllm_ascend/compilation/acl_graph.py` — `set_graph_params()` 与图捕获/回放核心流程
  - `vllm_ascend/ascend_config.py` — `cudagraph_capture_sizes` 读取处

### 背景与原理

decode 阶段每生成 1 个 token 都要跑一遍完整 forward，且每步 batch 形状相同。

```
Eager 模式 (CPU 逐个下发算子):
CPU:  [launch算子1]→[launch算子2]→…→[launch算子200]→ GPU 才开算
      └── 每个算子有几十 μs 的 CPU 调度/launch 开销 ──┘
      小模型/小 batch 时, CPU 下发时间甚至超过 GPU 计算时间!

图模式 (录制一次, 回放 N 次):
第1次: [录下全部 200 个算子的执行序列] → 存成一张"图"
之后:  [整图一次回放] → CPU 只发 1 条指令, GPU 连续执行
      └── 开销: 一次录制(启动时) + 显存存图, 换每步省下 CPU 调度 ──┘
      └── 限制: 只有"形状固定"的计算才能录 → 主要加速 decode;
          prefill 形状多变 → PIECEWISE 模式只录形状固定的部分
```

这就是启动命令里 `--enforce-eager`（强制 eager = 关掉图模式）牺牲掉的东西。

图模式要为每个 batch size 预先录一张图。`set_graph_params()` 会为 capture sizes 列表里的每一个 batch size 建一个图条目——所以 sizes 越长，预录的图越多，启动越慢、显存吃得越多，但运行时越不容易"miss 图、回退 eager"。

**牺牲了什么换来什么**：牺牲启动时间和显存，换来 decode 阶段每步的 CPU 调度开销消除（TPOT/ITL 下降）。

### 第 1 步：开关对比（不改代码）

```bash
# A组: eager（基线已跑）
# B组: PIECEWISE 图模式（去掉 enforce-eager）
EXTRA_ARGS="" TAG=graph_on bash 01_start_server.sh
TAG=graph_on bash 02_bench.sh 128 256 64 8
```

预期：TPOT / ITL 明显下降；TTFT 变化不大（图模式主要加速形状固定的 decode）。

### 第 2 步：加 1 行日志，观察 capture sizes

```bash
cd /workspace/vllm-ascend
git checkout -b exp3-aclgraph-sizes
```

在 `vllm_ascend/ascend_config.py` 的 `cudagraph_capture_sizes` 读取处（约 L1330）插入一行：

```python
capture_sizes = vc.compilation_config.cudagraph_capture_sizes
print(f"[EXP-ACLGRAPH] capture_sizes={capture_sizes}")   # ← 唯一新增的一行
capture_bound = max(capture_sizes) if capture_sizes else None
```

```bash
git add -A && git commit -m "exp3: log aclgraph capture sizes"
```

### 第 3 步：实验矩阵——sizes 集合对性能/启动的影响

```bash
# 默认集合: 观察启动日志里的 [EXP-ACLGRAPH] 和图捕获耗时
EXTRA_ARGS="" TAG=g_default bash 01_start_server.sh
grep "EXP-ACLGRAPH" /workspace/perf_test/serve_g_default.log
TAG=g_default bash 02_bench.sh 128 256 64 8

# 精简集合: 只录小 batch
EXTRA_ARGS="--compilation-config '{\"cudagraph_capture_sizes\": [1,2,4,8]}'" TAG=g_small bash 01_start_server.sh
TAG=g_small bash 02_bench.sh 128 256 64 8
```

对比三件事：启动耗时（日志时间戳）、TPOT（压测 json）、显存峰值（`npu-smi info`）。

### 预期学到的结论

| | 默认全量 capture | 精简 capture |
|---|---|---|
| 启动时间 | 长（要录 N 张图） | 短 |
| 显存 | 高（每张图占显存） | 低 |
| 高并发时 TPOT | 稳定（总有图可回放） | 大 batch 时 miss 图 → 回退 eager → 变慢 |

### 回滚

```bash
git checkout main          # 切回官方代码，实验都在各自分支上不丢
```

---

## 实验 4：投机解码（Speculative Sampling, ICML'23）

- 论文：https://arxiv.org/abs/2302.01318

### 背景与原理

decode 是 memory-bound 的：每步 forward 只为产出 1 个 token，显存带宽读整个权重和 KV cache，算力大量闲置。投机解码用"并行验证"把算力喂饱：

```
普通 decode (一次 forward 只出 1 个 token):
  [forward]→tok1 → [forward]→tok2 → [forward]→tok3   3 个 token = 3 次 forward

投机解码 (小模型起草, 大模型验证):
  draft 模型(快):   猜 tok1 tok2 tok3 tok4    ← 一次猜 4 个, 很便宜
  target 模型:      [一次 forward 同时验证 4 个] ← 并行验证, 算力被喂饱
  结果: 接受 tok1 tok2 tok3, 拒绝 tok4 → 按概率差修正
        3 个 token 只花了 1 次 target forward + 1 次便宜的 draft forward

  接受率 α 是命门: α 足够大才赚 (论文核心不等式)
  代码生成 token 可预测 → α 高 → 赚
  闲聊/高温采样      → α 低 → 亏
```

数学上，rejection sampling 机制保证接受的 token 严格服从 target model 的分布——输出分布无偏。

**牺牲了什么换来什么**：牺牲每次验证多算的开销（draft forward + 被拒绝 token 的计算），换来高接受率场景下 TPOT 数倍下降。

### 执行（配置级）

vllm-ascend 的投机解码以 EAGLE 路径为主，ngram 方法在 NPU 上需实测（报错则查当前版本支持矩阵）：

```bash
# 尝试 ngram 方法（vLLM 通用, NPU 上不一定完整支持）
EXTRA_ARGS="--speculative-config '{\"method\":\"ngram\",\"prompt_lookup_num\":4,\"num_speculative_tokens\":3}'" TAG=spec_ngram bash 01_start_server.sh
TAG=spec_ngram bash 02_bench.sh 128 256 32 4
```

看 `serve_spec_ngram.log` 里的 acceptance rate（接受率）：

- 接受率 > 50%：TPOT 应明显下降
- 接受率 < 30%：收益为负——这本身就是论文中 α 与猜测长度权衡关系的实证

### 验证标准

| 指标 | 预期 |
|---|---|
| TPOT | 随接受率上升而下降 |
| 输出分布 | 与不开投机等价（rejection sampling 保证数学无偏） |
| 吞吐 | 变化不大（省下的时间被 draft 开销抵消一部分） |

---

## 建议执行节奏

| 阶段 | 内容 | 产出 |
|---|---|---|
| 第 1 周 | 实验 0 + 实验 1 | 基线数据 + 互扰现象对比表 + batch 组成探针图 |
| 第 2 周 | 实验 2 | 共享前缀场景 TTFT 降幅报告 |
| 第 3 周 | 实验 3 | capture sizes 的"启动时间 × 显存 × TPOT"权衡结论 |
| 第 4 周 | 实验 4 | 接受率与 TPOT 的关系曲线 |

每个实验的存档物：`results/*.json`（数据）+ 探针 CSV/图（现象）+ git commit（改动）+ 一段自己的量化结论。这套"改动 → 数据 → 结论"的证据链，就是简历上"性能优化实践"的全部素材。

## 进阶方向（做完实验 0-4 后）

1. 把实验 1 的临时探针改造成正式功能：在 `vllm_ascend/envs.py` 注册 `VLLM_ASCEND_ENABLE_BATCH_PROBE` 环境变量 + 独立观测模块 + 单元测试——走一遍仓库正式贡献流程
2. 给 `ascend_chunked_prefill_workspace_size()` 加环境变量覆盖入口，把写死的经验公式变成可调旋钮，系统性扫参
3. 毕业项目：PD 分离（DistServe, OSDI'24, https://arxiv.org/abs/2401.09670），8 卡拆 2P+6D，Mooncake KV 传输，参考 `examples/disaggregated_prefill_v1/` 下的现成示例

## 参考资料

- AI Infra 学习指南（草帽路飞）：https://caomaolufei.github.io/AIInfraGuide/
- vLLM/PagedAttention (SOSP'23)：https://arxiv.org/abs/2309.06180
- Sarathi-Serve (OSDI'24)：https://www.usenix.org/conference/osdi24/presentation/agrawal
- SGLang/RadixAttention (NeurIPS'24)：https://arxiv.org/abs/2312.07104
- Speculative Sampling (ICML'23)：https://arxiv.org/abs/2302.01318
- Orca / Continuous Batching (OSDI'22)：https://www.usenix.org/conference/osdi22/presentation/yu
- DistServe (OSDI'24)：https://arxiv.org/abs/2401.09670
- vLLM Ascend 官方文档：https://docs.vllm.ai/projects/ascend/en/latest/
