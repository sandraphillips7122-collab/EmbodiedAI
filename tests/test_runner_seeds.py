"""``embodied_core.rollout.runner`` 的 seed 逻辑测试。

为什么单独测 seed：本仓库所有成功率都是"固定 seed 可复现"的。
如果 ``_episode_seed`` 不够确定（例如依赖全局 random、或不同 seed 块之间撞车），
那么"同一个 checkpoint 换个 seed 块得到 0.393 vs 0.167"这种结论就无法解释 ——
到底是策略波动还是 seed 设计有 bug，就分不清了。
"""

from __future__ import annotations

import pytest

from embodied_core.rollout.runner import _episode_seed


def test_episode_seed_is_deterministic() -> None:
    """同样的 (seed, i) 必须永远给同一个值。"""
    for seed in (0, 1, 2, 10, 11, 12, 500_000):
        for i in range(20):
            assert _episode_seed(seed, i) == _episode_seed(seed, i)


def test_episode_seed_is_injective_within_block() -> None:
    """同一个 seed 块内，不同 episode 不能拿到相同 seed。"""
    for seed in (0, 1, 2):
        values = [_episode_seed(seed, i) for i in range(200)]
        assert len(set(values)) == len(values), f"seed 块 {seed} 内出现重复"


def test_episode_seed_blocks_do_not_overlap() -> None:
    """不同 seed 块之间不能重叠（否则"块"就没有意义了）。"""
    blocks = {s: {_episode_seed(s, i) for i in range(200)} for s in (0, 1, 2, 10, 11, 12)}
    seeds_sorted = sorted(blocks)
    for a_idx, a in enumerate(seeds_sorted):
        for b in seeds_sorted[a_idx + 1:]:
            assert not (blocks[a] & blocks[b]), f"seed 块 {a} 与 {b} 有重叠"


def test_episode_seed_non_negative() -> None:
    """seed 不能为负（部分环境对负 seed 行为不一致）。"""
    for seed in (0, 1, 12, 999):
        for i in range(50):
            assert _episode_seed(seed, i) >= 0


def test_episode_seed_changes_with_episode_index() -> None:
    """相邻 episode 的 seed 必须不同（否则每集初始状态一样，评测就退化了）。"""
    assert _episode_seed(0, 0) != _episode_seed(0, 1)


@pytest.mark.parametrize("n_blocks", [1, 3])
def test_same_seed_tuple_reproduces_same_sequence(n_blocks: int) -> None:
    """模拟"两次评测用同样的 seeds 元组"：必须得到完全一致的 seed 序列。"""
    seeds = tuple(range(100, 100 + n_blocks))
    first = [_episode_seed(s, i) for s in seeds for i in range(10)]
    second = [_episode_seed(s, i) for s in seeds for i in range(10)]
    assert first == second
