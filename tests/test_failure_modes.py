"""``embodied_core.metrics.failure_modes`` 的单元测试。

失败模式分类把粗粒度的 ``timeout`` 拆成可归因的类别。因为它直接支撑
"**60% 的失败是抓起来后滑落**"这个结论，所以每个分支都必须被显式测到 ——
否则一旦分支条件写错（例如 ``elif`` 顺序调换），归因结论就会静默失真。

注意：这是**规则启发式**分类，不是人工标注。测试固定的是"规则按文档描述执行"，
不是"规则一定正确"。阈值定义在模块头部常量里，便于复核。
"""

from __future__ import annotations

import pytest

from embodied_core.metrics.failure_modes import (
    CUBE_MOVE_EPS,
    GOAL_NEAR_EPS,
    EpisodeSignals,
    classify_failure,
    summarize_failure_modes,
)


# --------------------------------------------------------------------------------------
# 成功
# --------------------------------------------------------------------------------------

def test_success_short_circuits_everything() -> None:
    """成功时无论信号多可疑，标签都必须是 success。"""
    weird = EpisodeSignals(
        ever_grasped=False,
        cube_displacement=0.0,
        final_goal_distance=99.0,
        steps=1,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(True, weird) == "success"


# --------------------------------------------------------------------------------------
# 四个失败分支
# --------------------------------------------------------------------------------------

def test_slip_when_grasped_but_far_from_goal() -> None:
    """夹住过、但最终离目标还远 → 抓取后滑落/丢失。"""
    s = EpisodeSignals(
        ever_grasped=True,
        cube_displacement=0.20,
        final_goal_distance=0.50,
        steps=150,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(False, s) == "slip"


def test_slip_when_grasped_but_cube_never_moved() -> None:
    """夹住过但方块几乎没动 → 仍归 slip（抓了但没带起来），不是 no_contact。"""
    s = EpisodeSignals(
        ever_grasped=True,
        cube_displacement=0.0,
        final_goal_distance=0.30,
        steps=150,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(False, s) == "slip"


def test_not_reached_when_moved_close_to_goal() -> None:
    """夹住过、方块被送到接近目标但差一点 → not_reached。"""
    s = EpisodeSignals(
        ever_grasped=True,
        cube_displacement=0.25,
        final_goal_distance=GOAL_NEAR_EPS / 2,  # 明确小于阈值
        steps=150,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(False, s) == "not_reached"


def test_collision_when_moved_without_grasping() -> None:
    """从未夹住却把方块推动了 → 碰撞/推飞。"""
    s = EpisodeSignals(
        ever_grasped=False,
        cube_displacement=CUBE_MOVE_EPS * 10,
        final_goal_distance=0.40,
        steps=150,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(False, s) == "collision"


def test_timeout_when_nothing_moved_and_steps_exhausted() -> None:
    """没夹住、没推动、步数用尽 → timeout。"""
    s = EpisodeSignals(
        ever_grasped=False,
        cube_displacement=0.0,
        final_goal_distance=0.30,
        steps=150,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(False, s) == "timeout"


def test_no_contact_when_episode_ended_early_without_movement() -> None:
    """未夹住、未移动、且**没有**用尽步数（提前结束）→ no_contact。"""
    s = EpisodeSignals(
        ever_grasped=False,
        cube_displacement=0.0,
        final_goal_distance=0.30,
        steps=17,
        max_steps=150,
        terminated=True,
        truncated=False,
    )
    assert classify_failure(False, s) == "no_contact"


# --------------------------------------------------------------------------------------
# 阈值边界
# --------------------------------------------------------------------------------------

def test_movement_threshold_is_strict_greater_than() -> None:
    """位移恰好等于阈值时不算"被移动过"（实现用的是严格大于）。"""
    at_threshold = EpisodeSignals(
        ever_grasped=False,
        cube_displacement=CUBE_MOVE_EPS,
        steps=150,
        max_steps=150,
        truncated=True,
    )
    just_over = EpisodeSignals(
        ever_grasped=False,
        cube_displacement=CUBE_MOVE_EPS + 1e-6,
        steps=150,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(False, at_threshold) == "timeout"
    assert classify_failure(False, just_over) == "collision"


def test_goal_distance_none_is_treated_as_far() -> None:
    """``final_goal_distance`` 缺失时不能误判成"已接近目标"。"""
    s = EpisodeSignals(
        ever_grasped=True,
        cube_displacement=0.2,
        final_goal_distance=None,
        steps=150,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(False, s) == "slip"


def test_not_reached_requires_both_moved_and_near_goal() -> None:
    """只"接近目标"但方块没被移动过，不应算 not_reached（该走 slip 分支）。"""
    s = EpisodeSignals(
        ever_grasped=True,
        cube_displacement=0.0,          # 没移动
        final_goal_distance=0.001,      # 很近
        steps=150,
        max_steps=150,
        truncated=True,
    )
    assert classify_failure(False, s) == "slip"


# --------------------------------------------------------------------------------------
# 统计汇总
# --------------------------------------------------------------------------------------

def test_summarize_counts_and_drops_absent_labels() -> None:
    """统计各标签次数，且**只返回出现过的标签**（便于对照阅读）。"""
    labels = ["success", "slip", "slip", "collision", "timeout"]
    counts = summarize_failure_modes(labels)
    assert counts == {"success": 1, "slip": 2, "collision": 1, "timeout": 1}
    assert "no_contact" not in counts
    assert sum(counts.values()) == len(labels)


def test_summarize_preserves_declared_order() -> None:
    """输出顺序固定（success → no_contact → slip → not_reached → collision → timeout），
    这样两个实验的失败模式字典可以直接对比。"""
    counts = summarize_failure_modes(["timeout", "success", "collision", "slip"])
    assert list(counts.keys()) == ["success", "slip", "collision", "timeout"]


def test_summarize_empty_list() -> None:
    assert summarize_failure_modes([]) == {}


@pytest.mark.parametrize(
    "signals,expected",
    [
        (EpisodeSignals(ever_grasped=True, cube_displacement=0.3, final_goal_distance=0.5), "slip"),
        (EpisodeSignals(ever_grasped=True, cube_displacement=0.3, final_goal_distance=0.01), "not_reached"),
        (EpisodeSignals(ever_grasped=False, cube_displacement=0.3), "collision"),
    ],
)
def test_classify_is_deterministic_across_repeated_calls(
    signals: EpisodeSignals, expected: str
) -> None:
    """同一个信号反复分类必须给同样结果（纯函数，无隐藏状态）。"""
    for _ in range(5):
        assert classify_failure(False, signals) == expected
