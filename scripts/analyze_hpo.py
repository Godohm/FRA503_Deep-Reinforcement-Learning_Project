"""Build the markdown HPO report from results/hpo/hpo_results.csv.

Per-algo OAT analysis: for each variation, compute delta vs the baseline row
in (Sharpe, total_return, MDD, n_trades) and rank parameters by absolute
impact on Sharpe. Also picks a "recommended config" per algo: the variant
with the highest Sharpe (with total_return as tie-break).

Outputs
-------
    results/hpo/hpo_report.md
    results/hpo/hpo_recommended.yaml   (per-algo dict of recommended overrides)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


# Hand-written copy text explaining each rollout knob for the report.
PARAM_DOC: dict[str, dict[str, str]] = {
    "ddqn": {
        "lr": "Adam learning rate for the online Q-network. "
              "High lr → faster learning but unstable bootstrapping; low lr → "
              "more stable but slow to absorb new transitions.",
        "eps_decay_steps": "Length (in action-steps) of the linear ε decay from 1.0 → 0.05. "
                           "Short decay → exploit sooner (good if env signal is clear); "
                           "long decay → keep exploring (avoids early policy collapse).",
        "target_update_freq": "Hard-sync interval (in learner updates) between online and target Q-net. "
                              "Smaller → target tracks online closely, faster TD propagation, "
                              "but loses the stabilising lag; larger → slower convergence, more stable.",
        "gamma": "Discount factor for future reward. "
                 "Intraday sessions reset bankroll each day → 0.95 is defensible (shorter horizon); "
                 "0.99 weights end-of-day reward roughly the same as next-bar reward.",
    },
    "a2c": {
        "n_steps": "Number of env steps collected per gradient update (n-step return window). "
                   "Small → frequent updates, high variance; large → smoother advantage estimates "
                   "but stale on-policy data.",
        "learning_rate": "Adam learning rate for both actor and critic. "
                         "A2C is sensitive — too high can collapse the policy to a single action.",
        "ent_coef": "Entropy bonus weight in the loss. "
                    "Higher → forces stochastic policy (more exploration); "
                    "zero → policy can become deterministic early.",
    },
    "ppo": {
        "n_steps": "Rollout buffer size per policy update. "
                   "Small → many updates with stale-but-fresh data; large → fewer updates per total budget "
                   "but each update sees a more representative batch.",
        "n_epochs": "Number of passes over each rollout. "
                    "More epochs → re-use data harder, risk overfitting to a single rollout; "
                    "fewer → less sample-efficient.",
        "clip_range": "PPO importance-ratio clip. "
                      "Tighter (e.g. 0.1) → more conservative updates, less policy drift; "
                      "looser (0.3) → faster but riskier policy moves.",
        "learning_rate": "Adam learning rate. "
                         "PPO tolerates a moderate range; very high lr defeats the clip stabilisation.",
    },
}


def _fmt(v) -> str:
    if v is None or pd.isna(v):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta_str(delta: float, *, fmt: str = ".4f") -> str:
    if pd.isna(delta):
        return ""
    sign = "+" if delta > 0 else ""
    return f"{sign}{format(delta, fmt)}"


def _analyse_algo(df_algo: pd.DataFrame) -> tuple[str, dict, dict]:
    """Return (markdown_section, recommended_overrides, ranking_dict) for one algo."""
    algo = df_algo["algo"].iloc[0]
    if "baseline" not in df_algo["variant"].values:
        return f"\n## {algo.upper()}\n\n(no baseline row found — skipping)\n", {}, {}

    base = df_algo.loc[df_algo["variant"] == "baseline"].iloc[0]
    metric_cols = ["sharpe", "total_return", "max_drawdown_pct", "n_trades", "exposure_time"]
    rows: list[dict] = []
    for _, r in df_algo.iterrows():
        rows.append({
            "variant": r["variant"],
            "param_changed": r["param_changed"],
            "param_value": r["param_value"],
            "sharpe": float(r["sharpe"]),
            "total_return": float(r["total_return"]),
            "max_drawdown_pct": float(r["max_drawdown_pct"]),
            "n_trades": int(r["n_trades"]),
            "exposure_time": float(r["exposure_time"]),
            "d_sharpe": float(r["sharpe"]) - float(base["sharpe"]),
            "d_return": float(r["total_return"]) - float(base["total_return"]),
            "d_trades": int(r["n_trades"]) - int(base["n_trades"]),
            "d_mdd": float(r["max_drawdown_pct"]) - float(base["max_drawdown_pct"]),
        })
    out = pd.DataFrame(rows)

    # Sensitivity ranking: |Δ Sharpe| from baseline, excluding the baseline row itself.
    variations_only = out[out["variant"] != "baseline"].copy()
    variations_only["abs_d_sharpe"] = variations_only["d_sharpe"].abs()
    sensitivity = variations_only.sort_values("abs_d_sharpe", ascending=False)

    # Recommended config = max Sharpe (tie-break: max total_return).
    rec = out.sort_values(by=["sharpe", "total_return"], ascending=False).iloc[0]
    rec_overrides: dict[str, Any] = {}
    if rec["variant"] != "baseline":
        # Coerce numpy/pandas scalars to plain Python so yaml.safe_dump works.
        raw_val = rec["param_value"]
        try:
            f = float(raw_val)
            coerced: Any = int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            coerced = str(raw_val)
        rec_overrides = {str(rec["param_changed"]): coerced}

    lines = [f"\n## {algo.upper()}\n"]
    lines.append("### Parameters explained\n")
    for k, doc in PARAM_DOC.get(algo, {}).items():
        lines.append(f"- **`{k}`** — {doc}")
    lines.append("")
    lines.append("### Variation results (vs baseline)\n")
    lines.append(
        "| variant | param changed | value | Sharpe | Δ Sharpe | total_return | Δ return | MDD% | Δ MDD | trades | Δ trades |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for _, r in out.iterrows():
        lines.append(
            f"| {r['variant']} | {r['param_changed']} | {r['param_value']} | "
            f"{_fmt(r['sharpe'])} | {_delta_str(r['d_sharpe'])} | "
            f"{_fmt(r['total_return'])} | {_delta_str(r['d_return'])} | "
            f"{_fmt(r['max_drawdown_pct'])} | {_delta_str(r['d_mdd'])} | "
            f"{int(r['n_trades'])} | {_delta_str(r['d_trades'], fmt='+d') if r['d_trades']!=0 else '0'} |"
        )
    lines.append("")

    lines.append("### Sensitivity ranking (by |Δ Sharpe| vs baseline)\n")
    lines.append("| rank | param | value | Δ Sharpe | Δ return | direction |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for i, (_, r) in enumerate(sensitivity.iterrows(), start=1):
        direction = "↑ helped" if r["d_sharpe"] > 0 else ("↓ hurt" if r["d_sharpe"] < 0 else "≈")
        lines.append(
            f"| {i} | {r['param_changed']} | {r['param_value']} | "
            f"{_delta_str(r['d_sharpe'])} | {_delta_str(r['d_return'])} | {direction} |"
        )
    lines.append("")

    # Discussion: most impactful + recommended
    if not sensitivity.empty:
        top = sensitivity.iloc[0]
        lines.append("### Discussion\n")
        lines.append(
            f"- Most impactful single change in this sweep: **`{top['param_changed']}` → "
            f"{top['param_value']}** (Δ Sharpe = {_delta_str(top['d_sharpe'])}). "
        )
        helping = sensitivity[sensitivity["d_sharpe"] > 0]
        hurting = sensitivity[sensitivity["d_sharpe"] < 0]
        if not helping.empty:
            tops = ", ".join(f"`{r['param_changed']}={r['param_value']}`" for _, r in helping.iterrows())
            lines.append(f"- Improved over baseline: {tops}.")
        if not hurting.empty:
            tops = ", ".join(f"`{r['param_changed']}={r['param_value']}`" for _, r in hurting.iterrows())
            lines.append(f"- Worse than baseline: {tops}.")

        if rec["variant"] == "baseline":
            lines.append("- **Recommendation:** keep the baseline — no variation improved Sharpe in this short sweep.")
        else:
            lines.append(
                f"- **Recommendation:** apply `{rec['param_changed']} = {rec['param_value']}` for the M8 full run."
            )
        lines.append(
            "- *Caveat:* 20k steps with single seed=42; signal is directional, not statistically robust. "
            "Validate the recommended config across the M8 seed sweep before trusting any ranking."
        )
        lines.append("")

    return "\n".join(lines), rec_overrides, {"sensitivity": sensitivity.to_dict("records")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HPO markdown report.")
    parser.add_argument("--csv", default="results/hpo/hpo_results.csv")
    parser.add_argument("--out-dir", default="results/hpo")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file():
        print(f"[error] CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[error] CSV is empty.")
        return 1

    pieces = [
        "# HPO short-sweep report — DDQN / A2C / PPO\n",
        "Generated from `results/hpo/hpo_results.csv` "
        f"({len(df)} runs across {df['algo'].nunique()} algorithms).\n",
        "**Design:** one-at-a-time (OAT) variation around a per-algo baseline, "
        "20k env steps per run, seed=42, eval every 5k steps. "
        "Each variation flips ONE parameter from the baseline, so each row's "
        "Δ Sharpe is attributable to that single change.\n",
        "**Caveats:**\n"
        "- 20k steps is a short horizon — agents are NOT converged; rankings are directional.\n"
        "- Single seed (42) → seed-induced variance is not measured here. The M8 full run "
        "  (200k steps × 3 seeds) is the source of statistical claims.\n"
        "- Test split is Dec 2024 only (20 sessions). Test-Sharpe is noisy on small windows.\n",
    ]

    recommended: dict[str, dict[str, str]] = {}
    for algo, df_algo in df.groupby("algo", sort=False):
        section, rec_overrides, _rank = _analyse_algo(df_algo.reset_index(drop=True))
        pieces.append(section)
        if rec_overrides:
            recommended[algo] = rec_overrides

    pieces.append("\n---\n## Suggested M8 overrides\n")
    if recommended:
        pieces.append("Per-algo, apply these one-line overrides to the relevant config "
                      "before running `scripts/run_experiment_1.py`:\n")
        for algo, overrides in recommended.items():
            pieces.append(f"- **{algo.upper()}** (`configs/{ {'ddqn':'dqn','a2c':'a2c','ppo':'ppo'}[algo] }.yaml`): "
                          f"set under the `{algo if algo!='ddqn' else 'dqn'}:` section: `{overrides}`")
    else:
        pieces.append("No algo's HPO sweep beat its own baseline. Keep all baselines.\n")

    md_path = out_dir / "hpo_report.md"
    md_path.write_text("\n".join(pieces), encoding="utf-8")

    rec_path = out_dir / "hpo_recommended.yaml"
    with rec_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(recommended or {}, f, sort_keys=False)

    print(f"[done] wrote:")
    print(f"  - {md_path}")
    print(f"  - {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
