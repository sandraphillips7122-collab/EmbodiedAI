"""策略封装（EmbodiedCore · rollout）。

统一接口让项目 A（ACT / Diffusion Policy）与项目 B（SmolVLA）能用**同一个评测器**，
这是两个项目结果可直接对照的前提。

约定：``predict`` 返回一个 shape 为 ``(action_dim,)`` 的 numpy 数组（单环境）。
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    """所有策略（学习到的或占位的）都必须满足的接口。"""

    name: str

    def reset(self, *, seed: int | None = None) -> None:
        """在每个 episode 开始时调用，清空内部状态（如动作队列）。"""

    def predict(self, obs: Any, instruction: str | None = None) -> np.ndarray:
        """给定观测（和可选的語言指令）返回一个动作。"""


class ZeroPolicy:
    """零动作基线：什么都不做，成功率应接近 0。用于验证评测器本身正确。"""

    name = "zero"

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def reset(self, *, seed: int | None = None) -> None:
        return None

    def predict(self, obs: Any, instruction: str | None = None) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


class RandomPolicy:
    """随机动作基线：同样应接近 0 成功率，但能覆盖整个动作空间。"""

    name = "random"

    def __init__(self, action_space, seed: int = 0):
        self.action_space = action_space
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def predict(self, obs: Any, instruction: str | None = None) -> np.ndarray:
        return np.asarray(self.action_space.sample(), dtype=np.float32)


def build_policy(name: str, *, action_space, action_dim: int, seed: int = 0) -> Policy:
    """按名字构造占位策略（Phase 1 用于验证评测器；Phase 2 起替换为学习到的策略）。"""
    if name == "zero":
        return ZeroPolicy(action_dim)
    if name == "random":
        return RandomPolicy(action_space, seed=seed)
    raise ValueError(f"未知的占位策略 {name!r}；可选 'zero' | 'random'")


__all__ = ["Policy", "ZeroPolicy", "RandomPolicy", "build_policy"]
