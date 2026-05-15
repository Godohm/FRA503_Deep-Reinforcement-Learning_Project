"""Plot M8 cross-seed comparison for the 2025 dataset run.

Reads:
    results/<algo>/<run_prefix>_<algo>_seed<S>_steps<N>/{metrics.json, trade_log.csv, eval_log.csv}

Emits:
    results/m8_2025/equity_curves.png       — overlaid equity curves per (algo, seed) on test split
    results/m8_2025/equity_per_algo_mean.png — mean equity curve per algo (band = min/max across seeds)
    results/m8_2025/drawdown_curves.png     — drawdown curves
    results/m8_2025/sharpe_bars.png         — bar chart of Sharpe per (algo, seed) + per-algo mean
    results/m8_2025/mdd_bars.png            — bar chart of MDD per (algo, seed)
    results/m8_2025/eval_curves.png         — validation Sharpe over training (1 line per (algo, seed))
    results/m8_2025/comparison_table.csv    — flat table of all metrics
    results/m8_2025/m8_comparison.md        — markdown summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ALGO_ORDER = ["ddqn", "ddqn_expdecay", "a2c", "ppo"]
ALGO_COLOR = {
    "ddqn":          "#1f77b4",  # blue
    "ddqn_expdecay": "#9467bd",  # purple
    "a2c":           "#2ca02c",  # green
    "ppo":           "#d62728",  # red
}
ALGO_LABEL = {
    "ddqn":          "DDQN (linear)",
    "ddqn_expdecay": "DDQN (exponential)",
    "a2c":           "A2C",
    "ppo":           "PPO",
}


def _algo_save_root(algo: str) -> str:
    return "ddqn" if algo.startswith("ddqn") else algo


def _load_run(algo: str, seed: int, run_prefix: str, steps: int):
    rid = f"{run_prefix}_{algo}_seed{seed}_steps{steps}"
    rdir = REPO_ROOT / "results" / _algo_save_root(algo) / rid
    if not rdir.is_dir():
        return None
    out = {"algo": algo, "seed": seed, "run_id": rid, "dir": rdir}
    metrics_p = rdir / "metrics.json"
    if metrics_p.is_file():
        out["metrics"] = json.loads(metrics_p.read_text())
    trade_p = rdir / "trade_log.csv"
    if trade_p.is_file():
        out["trade_log"] = pd.read_csv(trade_p)
    eval_p = rdir / "eval_log.csv"
    if eval_p.is_file():
        out["eval_log"] = pd.read_csv(eval_p)
    return out


def _equity_from_trade_log(trade_log: pd.DataFrame, initial_equity: float) -> np.ndarray:
    if "equity" in trade_log.columns:
        return trade_log["equity"].to_numpy(dtype=float)
    # Fallback: reconstruct from net_pnl
    return initial_equity + trade_log["net_pnl"].cumsum().to_numpy(dtype=float)


def _drawdown(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / np.maximum(peak, 1e-9)


def _plot_equity_curves(runs, out_dir, initial_equity):
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in runs:
        if "trade_log" not in r:
            continue
        eq = _equity_from_trade_log(r["trade_log"], initial_equity)
        ax.plot(np.arange(len(eq)), eq,
                color=ALGO_COLOR[r["algo"]], alpha=0.55, lw=0.8,
                label=f"{ALGO_LABEL[r['algo']]} seed={r['seed']}")
    ax.axhline(initial_equity, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("step (test set, sequential)")
    ax.set_ylabel("equity (USD)")
    ax.set_title("M8 2025 — Equity curves on TEST split (Mar–Apr 2026)")
    # Dedupe legend so we don't get 12 lines
    seen = set()
    handles, labels = [], []
    for h, l in zip(*ax.get_legend_handles_labels()):
        algo = l.split(" seed=")[0]
        if algo not in seen:
            seen.add(algo)
            handles.append(h)
            labels.append(algo)
    ax.legend(handles, labels, loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "equity_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_equity_per_algo_mean(runs, out_dir, initial_equity):
    """For each algo, plot the median equity curve across seeds with a min/max band."""
    by_algo: dict[str, list[np.ndarray]] = {}
    for r in runs:
        if "trade_log" not in r:
            continue
        eq = _equity_from_trade_log(r["trade_log"], initial_equity)
        by_algo.setdefault(r["algo"], []).append(eq)

    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in ALGO_ORDER:
        eqs = by_algo.get(algo, [])
        if not eqs:
            continue
        L = min(len(e) for e in eqs)
        arr = np.vstack([e[:L] for e in eqs])
        med = np.median(arr, axis=0)
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        x = np.arange(L)
        ax.plot(x, med, color=ALGO_COLOR[algo], lw=1.4, label=ALGO_LABEL[algo])
        ax.fill_between(x, lo, hi, color=ALGO_COLOR[algo], alpha=0.15)
    ax.axhline(initial_equity, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("step (test set)")
    ax.set_ylabel("equity (USD)")
    ax.set_title("M8 2025 — Median equity per algo on TEST split (band = min/max across seeds)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "equity_per_algo_mean.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_drawdowns(runs, out_dir, initial_equity):
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in runs:
        if "trade_log" not in r:
            continue
        eq = _equity_from_trade_log(r["trade_log"], initial_equity)
        dd = _drawdown(eq)
        ax.plot(np.arange(len(dd)), dd * 100.0,
                color=ALGO_COLOR[r["algo"]], alpha=0.55, lw=0.8)
    ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("step (test set)")
    ax.set_ylabel("drawdown (%)")
    ax.set_title("M8 2025 — Drawdown curves on TEST split")
    handles = [plt.Line2D([0], [0], color=ALGO_COLOR[a], lw=1.4, label=ALGO_LABEL[a])
               for a in ALGO_ORDER if any(r["algo"] == a for r in runs)]
    ax.legend(handles=handles, loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "drawdown_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_metric_bars(runs, out_dir, metric_key: str, ylabel: str, fname: str):
    """Grouped bar chart: x = algo, bars within group = seeds."""
    by_algo: dict[str, list[tuple[int, float]]] = {}
    for r in runs:
        if "metrics" not in r:
            continue
        v = r["metrics"].get(metric_key)
        if v is None:
            continue
        by_algo.setdefault(r["algo"], []).append((r["seed"], float(v)))
    algos = [a for a in ALGO_ORDER if a in by_algo]
    if not algos:
        return None
    seeds = sorted({s for vals in by_algo.values() for s, _ in vals})
    n_algos = len(algos)
    n_seeds = len(seeds)
    width = 0.8 / max(n_seeds, 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_algos), 4.5))
    x = np.arange(n_algos)
    for i, seed in enumerate(seeds):
        ys = []
        for algo in algos:
            d = dict(by_algo[algo])
            ys.append(d.get(seed, np.nan))
        offset = (i - (n_seeds - 1) / 2.0) * width
        ax.bar(x + offset, ys, width, label=f"seed={seed}",
               color=plt.cm.tab10(i / max(n_seeds, 1)))
    # Plot mean line
    means = [np.nanmean([v for _, v in by_algo[a]]) for a in algos]
    ax.plot(x, means, "k_", markersize=18, markeredgewidth=2, label="mean")
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABEL[a] for a in algos], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(f"M8 2025 — {ylabel} per (algo, seed)")
    ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / fname
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_eval_curves(runs, out_dir):
    """Plot evaluation Sharpe over training steps (per (algo, seed))."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in runs:
        if "eval_log" not in r:
            continue
        df = r["eval_log"]
        if df.empty or "sharpe" not in df.columns:
            continue
        ax.plot(df["step"], df["sharpe"],
                color=ALGO_COLOR[r["algo"]], alpha=0.55, lw=0.9,
                label=f"{ALGO_LABEL[r['algo']]} seed={r['seed']}")
    ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("Sharpe on val split")
    ax.set_title("M8 2025 — Periodic val Sharpe during training")
    # Dedupe legend
    seen = set()
    handles, labels = [], []
    for h, l in zip(*ax.get_legend_handles_labels()):
        algo = l.split(" seed=")[0]
        if algo not in seen:
            seen.add(algo)
            handles.append(h)
            labels.append(algo)
    ax.legend(handles, labels, loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "eval_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _build_comparison_table(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        if "metrics" not in r:
            continue
        m = r["metrics"]
        rows.append({
            "algo":         r["algo"],
            "seed":         r["seed"],
            "run_id":       r["run_id"],
            "total_return": m.get("total_return"),
            "sharpe":       m.get("sharpe_ratio"),
            "sortino":      m.get("sortino_ratio"),
            "max_dd_pct":   m.get("max_drawdown_pct"),
            "n_trades":     m.get("n_trades"),
            "win_rate":     m.get("win_rate"),
            "exposure":     m.get("exposure_time"),
            "final_equity": m.get("final_equity"),
        })
    return pd.DataFrame(rows)


def _write_markdown(df: pd.DataFrame, out_dir: Path) -> Path:
    md = ["# M8 2025 — Cross-Seed Comparison (TEST split)\n"]
    md.append(f"**Runs collected:** {len(df)} (TEST split = Mar–Apr 2026, 44 sessions).\n")
    if df.empty:
        md.append("\n(no runs found)\n")
        p = out_dir / "m8_comparison.md"
        p.write_text("\n".join(md), encoding="utf-8")
        return p

    md.append("## Per-(algo, seed)\n")
    md.append("| algo | seed | total_return | Sharpe | Sortino | MDD% | trades | win_rate | exposure | final_equity |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        md.append(
            f"| {ALGO_LABEL.get(r['algo'], r['algo'])} | {r['seed']} | "
            f"{r['total_return']:+.4f} | {r['sharpe']:+.3f} | {r['sortino']:+.3f} | "
            f"{r['max_dd_pct']:.4f} | {int(r['n_trades']) if pd.notna(r['n_trades']) else '—'} | "
            f"{r['win_rate']:.3f} | {r['exposure']:.3f} | ${r['final_equity']:,.0f} |"
        )

    md.append("\n## Cross-seed mean ± std\n")
    md.append("| algo | mean total_return | mean Sharpe | mean MDD% | mean trades |")
    md.append("|---|---|---|---|---|")
    for algo in ALGO_ORDER:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        md.append(
            f"| {ALGO_LABEL[algo]} | "
            f"{sub['total_return'].mean():+.4f} ± {sub['total_return'].std():.4f} | "
            f"{sub['sharpe'].mean():+.3f} ± {sub['sharpe'].std():.3f} | "
            f"{sub['max_dd_pct'].mean():.4f} ± {sub['max_dd_pct'].std():.4f} | "
            f"{sub['n_trades'].mean():.0f} ± {sub['n_trades'].std():.0f} |"
        )

    p = out_dir / "m8_comparison.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot M8 2025 cross-seed comparison.")
    parser.add_argument("--algos", nargs="+", default=ALGO_ORDER,
                        choices=ALGO_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--run-prefix", default="m8_2025")
    parser.add_argument("--out-dir", default="results/m8_2025")
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    missing = []
    for algo in args.algos:
        for seed in args.seeds:
            r = _load_run(algo, seed, args.run_prefix, args.steps)
            if r is None:
                missing.append((algo, seed))
            else:
                runs.append(r)

    print(f"[load] runs found: {len(runs)}, missing: {len(missing)}")
    if missing:
        for a, s in missing:
            print(f"  - missing: algo={a} seed={s}")
    if not runs:
        print("[fatal] no runs found; aborting.")
        return 1

    print("[plot] equity curves...")
    _plot_equity_curves(runs, out_dir, args.initial_equity)
    print("[plot] equity per-algo mean...")
    _plot_equity_per_algo_mean(runs, out_dir, args.initial_equity)
    print("[plot] drawdowns...")
    _plot_drawdowns(runs, out_dir, args.initial_equity)
    print("[plot] Sharpe bars...")
    _plot_metric_bars(runs, out_dir, "sharpe_ratio", "Sharpe", "sharpe_bars.png")
    print("[plot] MDD bars...")
    _plot_metric_bars(runs, out_dir, "max_drawdown_pct", "Max Drawdown (%)", "mdd_bars.png")
    print("[plot] total_return bars...")
    _plot_metric_bars(runs, out_dir, "total_return", "Total Return", "total_return_bars.png")
    print("[plot] eval curves...")
    _plot_eval_curves(runs, out_dir)

    df = _build_comparison_table(runs)
    csv_p = out_dir / "comparison_table.csv"
    df.to_csv(csv_p, index=False)
    print(f"[done] {csv_p}")

    md_p = _write_markdown(df, out_dir)
    print(f"[done] {md_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
