"""失败模式分类（EmbodiedCore · metrics）。

把 rollout 的失败从当前的粗粒度 ``timeout`` 细化为**可归因的类别**，依据是轨迹里的几何信号
（ManiSkill 的 ``obs.extra`` 与 ``info``），而不是事后猜测：

| 标签 | 判据（启发式，阈值写在下方常量里） |
|---|---|
| `success` | 环境返回 `info["success"]` |
| `no_contact` | 全程从未接触/夹住目标（`is_grasped` 从未为真，且方块位移很小） |
| `slip` | 曾经夹住（`is_grasped` 为真）但最后没有成功 → 抓取后滑落/丢失 |
| `not_reached` | 夹住了且方块被移动过，但最终未到达目标位（`goal_pos` 距离仍大） |
| `collision` | 方块被推移过（位移大但从未被夹住）→ 多为碰撞/推飞 |
| `timeout` | 以上都不满足且步数用尽 |
| `unknown` | 证据不足 |

**重要**：这是**启发式归因**，阈值集中在此文件便于复核；报告中必须声明它是规则分类而非人工标注。
调用方（评测循环）负责在每个 step 收集 `signals` 序列，然后调用 :func:`classify_failure`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 判据阈值（单位与 ManiSkill 一致：米）
CUBE_MOVE_EPS = 0.01      # 方块位移超过 1cm 视为「被移动过」
GOAL_NEAR_EPS = 0.05      # 距目标 5cm 内视为「已接近目标」（成功阈值是 0.025）


@dataclass
class EpisodeSignals:
    """一个 episode 过程中与失败归因相关的信号。"""

    ever_grasped: bool = False
    cube_displacement: float = 0.0          # 方块初始→最终位移（米）
    final_goal_distance: float | None = None  # 最终方块到目标的距离（米）
    steps: int = 0
    max_steps: int = 0
    terminated: bool = False
    truncated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def classify_failure(success: bool, s: EpisodeSignals) -> str:
    """把一次 rollout 归入一个失败模式标签（成功时返回 ``"success"``）。"""
    if success:
        return "success"

    moved = s.cube_displacement > CUBE_MOVE_EPS
    near_goal = s.final_goal_distance is not None and s.final_goal_distance < GOAL_NEAR_EPS

    if s.ever_grasped:
        # 夹住过：如果最后没成功，要么滑落，要么没送到目标
        return "not_reached" if (moved and near_goal) else "slip"
    if moved:
        # 完全没夹住却把方块推动了 → 碰撞 / 推飞
        return "collision"
    if s.truncated or s.steps >= s.max_steps:
        return "timeout"
    if not moved and not s.truncated:
        return "no_contact"
    return "unknown"


def summarize_failure_modes(labels: list[str]) -> dict[str, int]:
    """统计各失败模式出现的次数（保持确定顺序，便于对照）。"""
    order = ("success", "no_contact", "slip", "not_reached", "collision", "timeout", "unknown")
    counts = {k: 0 for k in order}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {k: v for k, v in counts.items() if v > 0}
