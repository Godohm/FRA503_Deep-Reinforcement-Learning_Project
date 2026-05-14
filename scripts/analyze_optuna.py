"""Generate dose-response plots + fANOVA importance + markdown report from
Optuna studies produced by ``scripts/optuna_search.py``.

Outputs
-------
    results/optuna/<algo>_dose_response.png   (one scatter+trend per parameter)
    results/optuna/<algo>_history.png         (objective vs trial# + best-so-far)
    results/optuna/<algo>_importance.png      (fANOVA bar chart)
    results/optuna/optuna_report.md           (consolidated report)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402


# Hand-written explanations of each tunable for the report write-up.
PARAM_DOC: dict[str, dict[str, str]] = {
    "ddqn": {
        "lr": "Adam step-size for the online Q-net. Too high → unstable bootstrapped TD targets; too low → never absorbs new transitions in the trial budget.",
        "gamma": "Discount on future reward. Intraday bankroll resets each session, so the effective horizon is short — values near 0.95 are theoretically defensible.",
        "eps_decay_steps": "Length (in agent action-steps) of linear ε decay 1.0 → 0.05. Short → exploit early (good if env signal is clear); long → keeps exploring.",
        "target_update_freq": "Hard-sync interval (in learner steps) between online and target Q-net. Smaller → target tracks online closely (fast but less stable).",
        "batch_size": "Replay minibatch size. Larger → smoother gradient, slower wall-time per step.",
        "min_buffer_to_learn": "Wait this many transitions before the first gradient update. Higher = warmer cold-start, but burns exploration budget.",
    },
    "a2c": {
        "learning_rate": "Adam step-size shared by actor and critic. A2C is notoriously sensitive — too high can collapse the policy to a single action.",
        "n_steps": "n-step return window per gradient update (rollout length). Small → frequent, noisy updates; large → smoother advantage but stale data.",
        "gamma": "Discount factor. See DDQN row.",
        "ent_coef": "Entropy bonus in the loss. Higher → forces stochastic policy (more exploration); zero → policy can collapse to deterministic early.",
        "vf_coef": "Weight of the value-function loss vs the policy gradient loss.",
        "gae_lambda": "Bias/variance trade-off in GAE. 1.0 = unbiased Monte-Carlo, 0.95 = standard smoothing.",
    },
    "ppo": {
        "learning_rate": "Adam step-size. PPO tolerates a moderate range; very high lr defeats the clip stabilisation.",
        "n_steps": "Rollout buffer size per policy update. Small → many updates per total step budget; large → fewer but each update sees more data.",
        "batch_size": "Minibatch within each rollout pass.",
        "n_epochs": "Passes over each rollout. More epochs → re-use data hard (overfit risk); fewer → less sample-efficient.",
        "gamma": "Discount factor. See DDQN row.",
        "clip_range": "PPO importance-ratio clip ε. Tight (0.1) → conservative; loose (0.3) → faster but riskier policy moves.",
        "ent_coef": "Entropy bonus, see A2C row.",
        "vf_coef": "Value-loss weight, see A2C row.",
        "gae_lambda": "GAE smoothing.",
    },
}


LOG_PARAMS = {"lr", "learning_rate", "ent_coef"}


_ALGO_PREFIXES = ("ddqn", "a2c", "ppo")


def _algo_of_tag(tag: str) -> str:
    """A *tag* is an algo name optionally suffixed (e.g. 'ddqn_expdecay').
    Return the underlying algorithm name for PARAM_DOC / sampler lookups."""
    for a in _ALGO_PREFIXES:
        if tag == a or tag.startswith(a + "_"):
            return a
    return tag  # fall back: caller will likely fail loudly later


def _label_of_tag(tag: str) -> str:
    """Human-readable section title. 'ddqn_expdecay' → 'DDQN — expdecay'."""
    algo = _algo_of_tag(tag)
    if tag == algo:
        return algo.upper()
    suffix = tag[len(algo) + 1:]   # strip 'algo_'
    return f"{algo.upper()} — {suffix}"


def _load_study(tag: str, storage_dir: Path) -> optuna.Study | None:
    db = storage_dir / f"{tag}_study.db"
    if not db.is_file():
        return None
    return optuna.load_study(
        study_name=f"hpo_{tag}",
        storage=f"sqlite:///{db}",
    )


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and t.value > -1.0e5]


def _plot_history(study: optuna.Study, algo: str, out_dir: Path) -> Path:
    trials = _completed_trials(study)
    if not trials:
        return out_dir / "(no trials)"
    idx = [t.number for t in trials]
    vals = np.array([t.value for t in trials], dtype=float)
    cummax = np.maximum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(idx, vals, alpha=0.6, s=30, label="trial Sharpe")
    ax.plot(idx, cummax, "g-", lw=1.5, label="best-so-far")
    ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("trial #")
    ax.set_ylabel("Sharpe (objective)")
    ax.set_title(f"{algo.upper()} — Optuna optimisation history (n={len(trials)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{algo}_history.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_dose_response(study: optuna.Study, algo: str, out_dir: Path) -> Path:
    trials = _completed_trials(study)
    if not trials:
        return out_dir / "(no trials)"
    params = sorted(trials[0].params.keys())
    n = len(params)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.2))
    axes = np.atleast_1d(axes).flatten()

    for i, p in enumerate(params):
        ax = axes[i]
        xs_raw = [t.params[p] for t in trials]
        ys = np.array([t.value for t in trials], dtype=float)

        # Categorical / discrete handling: if any value is non-numeric, treat as categorical.
        non_numeric = any(not isinstance(x, (int, float)) for x in xs_raw)
        if non_numeric:
            # Build a stable ordering and box-plot Sharpe per category.
            cats = sorted(set(xs_raw), key=lambda v: str(v))
            grouped = [ys[[j for j, v in enumerate(xs_raw) if v == c]] for c in cats]
            ax.boxplot(grouped, labels=[str(c) for c in cats], showmeans=True)
            ax.set_xlabel(p)
        else:
            xs = np.array(xs_raw, dtype=float)
            if p in LOG_PARAMS:
                ax.set_xscale("log")
                xn = np.log10(xs)
            else:
                xn = xs
            ax.scatter(xs, ys, alpha=0.6, s=30)
            # Polynomial-of-degree-2 trend if we have enough distinct values.
            if len(np.unique(xn)) >= 4:
                try:
                    coefs = np.polyfit(xn, ys, deg=2)
                    grid = np.linspace(xn.min(), xn.max(), 100)
                    pred = np.polyval(coefs, grid)
                    if p in LOG_PARAMS:
                        ax.plot(10**grid, pred, "r-", alpha=0.7, lw=1.2)
                    else:
                        ax.plot(grid, pred, "r-", alpha=0.7, lw=1.2)
                except Exception:
                    pass
            ax.set_xlabel(p)
        ax.set_ylabel("Sharpe")
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{algo.upper()} — Sharpe vs hyperparameter (n_trials={len(trials)})")
    fig.tight_layout()
    p = out_dir / f"{algo}_dose_response.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_importance(study: optuna.Study, algo: str,
                     out_dir: Path) -> tuple[Path | None, dict[str, float]]:
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"[{algo}] importance unavailable: {e}")
        return None, {}
    if not importance:
        return None, {}
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(items))))
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("fANOVA importance")
    ax.set_title(f"{algo.upper()} — parameter importance")
    fig.tight_layout()
    p = out_dir / f"{algo}_importance.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p, dict(items)


def _sharpe_slope_per_param(study: optuna.Study) -> dict[str, dict[str, float]]:
    """For each numeric param, fit a degree-1 polynomial Sharpe~param and
    return (slope, intercept, r2). Used for textual "increase A → improves B"
    discussion in the report.
    """
    out: dict[str, dict[str, float]] = {}
    trials = _completed_trials(study)
    if not trials:
        return out
    ys = np.array([t.value for t in trials], dtype=float)
    for p in trials[0].params.keys():
        xs_raw = [t.params[p] for t in trials]
        non_numeric = any(not isinstance(x, (int, float)) for x in xs_raw)
        if non_numeric:
            continue
        xs = np.array(xs_raw, dtype=float)
        if p in LOG_PARAMS:
            xs = np.log10(xs)
        if len(np.unique(xs)) < 3:
            continue
        slope, intercept = np.polyfit(xs, ys, deg=1)
        yhat = slope * xs + intercept
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        out[p] = {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
    return out


# --------------------------------------------------------------------------- markdown


def _render_algo_section(tag: str, study: optuna.Study,
                         importance: dict[str, float],
                         slopes: dict[str, dict[str, float]]) -> str:
    algo = _algo_of_tag(tag)
    label = _label_of_tag(tag)
    trials = _completed_trials(study)
    if not trials:
        return f"\n### {label}\n\n(no completed trials)\n"

    best = study.best_trial
    best_attrs = best.user_attrs

    lines: list[str] = [f"\n### {label}\n"]
    lines.append(f"- Trials completed: **{len(trials)} / {len(study.trials)}**")
    lines.append(f"- Best Sharpe: **{best.value:.3f}** (trial #{best.number})")
    lines.append(f"- Best total_return: {best_attrs.get('total_return', float('nan')):.4f}")
    lines.append(f"- Best MDD%: {best_attrs.get('max_drawdown_pct', float('nan')):.4f}")
    lines.append(f"- Best n_trades: {best_attrs.get('n_trades', 'n/a')}")
    lines.append("\n### Best hyperparameters")
    for k, v in best.params.items():
        lines.append(f"- `{k}`: **{v}**")

    lines.append("\n### Parameter importance (fANOVA)")
    if importance:
        lines.append("\n| rank | param | importance | direction (linear slope, R²) |")
        lines.append("| --- | --- | --- | --- |")
        for i, (k, v) in enumerate(importance.items(), start=1):
            s = slopes.get(k, {})
            if s:
                sign = "↑" if s["slope"] > 0 else "↓"
                dir_str = f"{sign} slope={s['slope']:+.3f}, R²={s['r2']:.2f}"
            else:
                dir_str = "(categorical)"
            lines.append(f"| {i} | `{k}` | {v:.4f} | {dir_str} |")
    else:
        lines.append("(fANOVA failed — typically too few trials.)")

    lines.append("\n### Parameter glossary (what each knob does)")
    for k in best.params.keys():
        doc = PARAM_DOC.get(algo, {}).get(k, "")
        if doc:
            lines.append(f"- **`{k}`** — {doc}")

    if slopes:
        lines.append("\n### Dose-response (linear) reading\n")
        lines.append("For numeric params with R² ≥ 0.1, the direction below is informative; "
                     "below that threshold the slope is dominated by noise.")
        for k, s in sorted(slopes.items(), key=lambda kv: kv[1]["r2"], reverse=True):
            if s["r2"] < 0.01:
                continue
            sign_word = "increasing" if s["slope"] > 0 else "decreasing"
            unit = " (log scale)" if k in LOG_PARAMS else ""
            lines.append(
                f"- `{k}`{unit}: Sharpe **trends {sign_word}** with the parameter "
                f"(slope = {s['slope']:+.3f}, R² = {s['r2']:.2f})."
            )

    lines.append("\n### Plots")
    lines.append(f"- Optimisation history: `{tag}_history.png`")
    lines.append(f"- Per-parameter dose-response: `{tag}_dose_response.png`")
    lines.append(f"- Importance bars: `{tag}_importance.png`")
    return "\n".join(lines)


def _render_variant_comparison(tag_studies: list[tuple[str, optuna.Study]]) -> str:
    """If two or more tags share an algo prefix, emit a side-by-side comparison
    of best Sharpe / best params. Useful for ablations like linear vs exponential
    ε-decay where the underlying algo is the same DDQN."""
    by_algo: dict[str, list[tuple[str, optuna.Study]]] = {}
    for tag, study in tag_studies:
        by_algo.setdefault(_algo_of_tag(tag), []).append((tag, study))

    sections: list[str] = []
    for algo, variants in by_algo.items():
        if len(variants) < 2:
            continue
        lines = [f"\n## 3a. Ablation — {algo.upper()} variants\n"]
        lines.append(
            "Each variant is a separate Optuna study over the *same search space* "
            "(see `scripts/optuna_search.py`) with one design knob held to a "
            "different value (encoded in the variant name). Best Sharpe is the "
            "single best trial across 50 random/TPE samples — read it as an "
            "*upper-bound* on what the design choice can deliver in 50k steps, "
            "not as a converged measurement.\n"
        )
        lines.append("| variant | best Sharpe | trial # | best total_return | n_trades | trials |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for tag, study in variants:
            trials = _completed_trials(study)
            if not trials:
                lines.append(f"| `{tag}` | (no trials) | — | — | — | 0 |")
                continue
            best = study.best_trial
            attrs = best.user_attrs
            label = tag if tag == algo else tag[len(algo) + 1:]
            lines.append(
                f"| `{label}` | **{best.value:.3f}** | #{best.number} | "
                f"{attrs.get('total_return', float('nan')):.4f} | "
                f"{attrs.get('n_trades', 'n/a')} | {len(trials)} |"
            )
        # Quick verdict.
        ranked = sorted(
            ((t, s) for t, s in variants if _completed_trials(s)),
            key=lambda ts: ts[1].best_trial.value, reverse=True,
        )
        if len(ranked) >= 2:
            winner_tag, winner = ranked[0]
            runner_tag, runner = ranked[1]
            delta = winner.best_trial.value - runner.best_trial.value
            w_label = winner_tag if winner_tag == algo else winner_tag[len(algo) + 1:]
            r_label = runner_tag if runner_tag == algo else runner_tag[len(algo) + 1:]
            lines.append(
                f"\n**Verdict:** `{w_label}` wins the best-of-50 with "
                f"Sharpe = {winner.best_trial.value:.3f}, beating `{r_label}` "
                f"(Sharpe = {runner.best_trial.value:.3f}) by Δ = {delta:+.3f}."
            )
        sections.append("\n".join(lines))
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse Optuna studies + emit report.")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="Deprecated alias for --tags. Kept for backward compat.")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Study tags to analyse, e.g. ddqn a2c ppo ddqn_expdecay. "
                             "Each tag <T> looks for {storage_dir}/<T>_study.db with "
                             "study_name=hpo_<T>.")
    parser.add_argument("--storage-dir", default="results/optuna")
    parser.add_argument("--out-dir", default="results/optuna")
    args = parser.parse_args()

    # Resolve tags from --tags or fall back to --algos.
    tags: list[str] = args.tags if args.tags is not None else (
        args.algos if args.algos is not None else ["ddqn", "a2c", "ppo"]
    )

    storage_dir = Path(args.storage_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[str] = [
        "# Bayesian HPO Report — Optuna TPE (2025–2026 Dataset)\n",

        "## 1. What is Optuna?\n",
        "**Optuna** is an open-source automatic hyperparameter optimisation (HPO) "
        "framework developed by Preferred Networks. Unlike grid search (exhaustive but "
        "exponential) or random search (simple but wasteful), Optuna builds a "
        "*probabilistic surrogate model* of the objective function and uses it to "
        "decide which hyperparameter configurations to evaluate next. This makes it "
        "far more sample-efficient: it finds good configurations in far fewer trials "
        "than grid or random search would need.\n",

        "### 1.1 How TPE Works\n",
        "The sampler used here is **TPE (Tree-structured Parzen Estimator)**, "
        "the default Optuna sampler and the same algorithm used in Hyperopt. "
        "It works as follows:\n\n"
        "1. **Startup phase** (first `n_startup_trials` trials, here 10): "
        "sample configurations *uniformly at random* to warm up the model — no "
        "prior knowledge yet.\n"
        "2. **Model phase** (from trial 11 onwards): split all past trials into two "
        "groups based on a quantile γ of the objective (Sharpe):\n"
        "   - **l(x)** — the *good* distribution: density model fitted on the top "
        "     γ fraction of configurations (high Sharpe).\n"
        "   - **g(x)** — the *bad* distribution: density model fitted on the bottom "
        "     1−γ fraction.\n"
        "   TPE then proposes the next configuration by maximising the ratio "
        "**l(x) / g(x)** — i.e. it favours regions that are likely under the "
        "*good* distribution and unlikely under the *bad* one.\n"
        "3. Each proposed configuration is then **evaluated** by training a fresh "
        "agent for 50 000 environment steps and measuring Sharpe on the *validation* "
        "split (Jan–Feb 2026). This Sharpe score is fed back to update l(x) and g(x).\n\n"
        "The key advantage over random search is that TPE concentrates evaluations in "
        "promising regions, converging to better configurations with the same budget.\n",

        "### 1.2 Study Setup\n",
        "| Setting | Value |\n"
        "| --- | --- |\n"
        "| Sampler | TPE (Optuna 4.8.0) |\n"
        "| Trials per study | 50 |\n"
        "| Steps per trial | 50 000 |\n"
        "| Startup (random) trials | 10 |\n"
        "| Objective metric | Sharpe ratio (annualised, on *validation* split) |\n"
        "| Seed | 42 |\n"
        "| Train data | Jan–Dec 2025 (257 sessions) |\n"
        "| Eval (HPO selection) | Jan–Feb 2026 *validation* split (40 sessions) |\n"
        "| Test (held out) | Mar–Apr 2026 — never touched during HPO |\n"
        "| Studies | DDQN-linear, DDQN-exponential, A2C, PPO |\n\n"
        "> **Why separate validation?** The old 2-way split used the test set to "
        "select checkpoints during training, leaking information into HPO. The "
        "3-way split fixes this: validation guides HPO, test is reserved for the "
        "final unbiased comparison reported in the paper.\n",

        "## 2. How to Read the Artefacts\n",
        "**Dose-response plot** (`<algo>_dose_response.png`) — for each sampled "
        "hyperparameter, a scatter of (parameter value, observed Sharpe) across all "
        "completed trials. A quadratic trend line is overlaid when ≥4 distinct values "
        "were sampled. Categorical parameters are shown as boxplots. The plot reveals "
        "*which parameter values correlate with higher Sharpe* and in which "
        "direction — useful for understanding the landscape the agent operates in.\n\n"
        "**fANOVA importance** (`<algo>_importance.png`) — the Hutter/Hoos fANOVA "
        "algorithm partitions variance in the Sharpe scores attributable to each "
        "hyperparameter (and their interactions). A parameter with importance = 0.48 "
        "explains ~48 % of the variation in Sharpe across all 50 trials. Note: "
        "*importance ≠ direction* — a highly important parameter might be one where "
        "the wrong value hurts badly. Consult the dose-response plot for sign.\n\n"
        "**History plot** (`<algo>_history.png`) — Sharpe vs trial number with the "
        "cumulative best-so-far overlaid. A rising best-so-far curve indicates TPE is "
        "successfully focusing on good regions. A flat curve after trial 10 suggests "
        "the search space is too wide or the objective is too noisy for TPE to learn "
        "from 50 trials.\n",

        "## 3. HPO Results\n",
        "The sections below cover each algorithm. The **Ablation** section (if "
        "present) compares variants of the same algorithm (e.g. linear vs exponential "
        "ε-decay for DDQN) trained under identical conditions.\n",
    ]

    tag_studies: list[tuple[str, optuna.Study]] = []
    for tag in tags:
        study = _load_study(tag, storage_dir)
        if study is None:
            pieces.append(f"\n### {_label_of_tag(tag)}\n\n(no study DB found at "
                          f"{storage_dir}/{tag}_study.db)\n")
            continue
        tag_studies.append((tag, study))

    # Variant ablation section emitted *before* the per-tag sections so readers
    # see the headline comparison first.
    ablation = _render_variant_comparison(tag_studies)
    if ablation:
        pieces.append(ablation)

    for tag, study in tag_studies:
        _plot_history(study, tag, out_dir)
        _plot_dose_response(study, tag, out_dir)
        _png_path, imp = _plot_importance(study, tag, out_dir)
        slopes = _sharpe_slope_per_param(study)
        pieces.append(_render_algo_section(tag, study, imp, slopes))

    md_path = out_dir / "optuna_report.md"
    md_path.write_text("\n".join(pieces), encoding="utf-8")
    print(f"[done] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
