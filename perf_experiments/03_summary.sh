#!/bin/bash
# 汇总对比所有压测结果: 从 results/*.json 提取 TTFT/TPOT/ITL 的 P50/P99, 打成一张表
#
# 用法:
#   bash 03_summary.sh                 # 汇总 /workspace/perf_test/results/ 下所有 json
#   bash 03_summary.sh cp_*            # 只汇总匹配模式的实验 (shell glob)

RESULTS_DIR=${1:-/workspace/perf_test/results}

python3 - "$RESULTS_DIR" <<'EOF'
import json, glob, sys, os

results_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/perf_test/results"
files = sorted(glob.glob(os.path.join(results_dir, "*.json")))

if not files:
    print(f"no result json found in {results_dir}")
    sys.exit(1)

def pick(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return "-"

header = (f"{'experiment':<42} {'ttft_p50':>9} {'ttft_p99':>9} "
          f"{'tpot_p50':>9} {'tpot_p99':>9} {'itl_p50':>8} {'itl_p99':>8} {'thru':>8}")
print(header)
print("-" * len(header))

for f in files:
    try:
        with open(f) as fp:
            d = json.load(fp)
    except (json.JSONDecodeError, OSError):
        continue
    name = os.path.basename(f).replace(".json", "")[:41]
    print(f"{name:<42} "
          f"{pick(d,'p50_ttft_ms','median_ttft_ms'):>9} "
          f"{pick(d,'p99_ttft_ms'):>9} "
          f"{pick(d,'p50_tpot_ms','median_tpot_ms'):>9} "
          f"{pick(d,'p99_tpot_ms'):>9} "
          f"{pick(d,'p50_itl_ms','median_itl_ms'):>8} "
          f"{pick(d,'p99_itl_ms'):>8} "
          f"{pick(d,'output_throughput','request_throughput'):>8}")
print()
print("(unit: ms for ttft/tpot/itl; throughput as reported by benchmark_serving)")
EOF
