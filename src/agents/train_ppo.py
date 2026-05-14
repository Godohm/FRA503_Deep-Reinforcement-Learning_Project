"""PPO training entrypoint via Stable-Baselines3."""
from __future__ import annotations

from typing import Any, Optional

from src.agents.sb3_common import train_sb3


def train_ppo(
    env_cfg: dict[str, Any],
    ppo_cfg: dict[str, Any],
    *,
    total_steps: Optional[int] = None,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    return train_sb3(
        algo="ppo",
        env_cfg=env_cfg,
        algo_cfg=ppo_cfg,
        total_steps=total_steps,
        run_id=run_id,
        seed=seed,
    )
