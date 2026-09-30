#!/usr/bin/env python3
"""画实验 1 探针 (probe_batch.csv) 的 batch 组成图。

用法:
    python3 plot_probe.py /workspace/perf_test/probe_batch.csv

前提: 已按教程在 model_runner_v1.py 的 execute_model 加了探针, 并带
MY_PROBE=1 启动服务跑过一轮压测。CSV 每行格式: 时间戳,请求数
"""
import csv
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("need matplotlib: pip install matplotlib")

def main(csv_path: str) -> None:
    ts, reqs = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                ts.append(float(row[0]))
                reqs.append(int(row[1]))
            except ValueError:
                continue  # skip header/corrupt lines

    if not ts:
        sys.exit(f"no data rows in {csv_path}")

    t0 = ts[0]
    xs = [t - t0 for t in ts]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(xs, reqs, width=0.08)
    ax.set_xlabel("time since first step (s)")
    ax.set_ylabel("requests per forward step")
    ax.set_title("execute_model batch composition per step (Sarathi Figure-1 style probe)")
    fig.tight_layout()
    out = csv_path.replace(".csv", ".png")
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
