"""CLI wrapper around src.agents.train_ppo.train_ppo."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.train_ppo import train_ppo  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 PPO on EURUSDIntradayTradingEnv.")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--ppo-config", default="configs/ppo.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    logger = logging.getLogger("train_ppo")

    env_cfg = load_config(args.env_config)
    ppo_cfg = load_config(args.ppo_config)

    if args.smoke:
        steps = int(ppo_cfg.get("training", {}).get("smoke_steps", 1000))
        logger.info("Smoke training enabled — %d steps", steps)
    else:
        steps = args.steps

    summary = train_ppo(env_cfg, ppo_cfg, total_steps=steps,
                        run_id=args.run_id, seed=args.seed)
    logger.info("Done. Summary: %s", summary)
    print(summary)


if __name__ == "__main__":
    main()
