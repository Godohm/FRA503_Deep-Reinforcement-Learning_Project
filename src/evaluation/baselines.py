"""Non-learned baseline policies used as reference in Experiment 3.

A "policy" here is a callable ``(obs: np.ndarray) -> int`` returning the
discrete action {0, 1, 2}. Baselines do not condition on the observation —
they simply target a fixed position every bar (the env's force-close override
still flattens at the last bar).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def long_only_policy(obs: np.ndarray) -> int:  # noqa: ARG001
    """Always request target position = +1 (long)."""
    return 2


def short_only_policy(obs: np.ndarray) -> int:  # noqa: ARG001
    """Always request target position = -1 (short)."""
    return 0


def flat_policy(obs: np.ndarray) -> int:  # noqa: ARG001
    """Always request target position = 0 (flat). Should yield zero PnL & zero cost."""
    return 1


BASELINES: dict[str, Callable[[np.ndarray], int]] = {
    "long_only": long_only_policy,
    "short_only": short_only_policy,
    "flat": flat_policy,
}
