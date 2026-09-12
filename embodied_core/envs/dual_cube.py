"""双色方块任务：**真正需要读指令**的最小语言条件任务。

## 为什么不能用"两个 goal 位置"

PickCube 的 `goal_site` 被注册进 `_hidden_objects`，而 ManiSkill 在每次取 sensor 观测前会执行
（`mani_skill/envs/sapien_env.py:601`）::

    for obj in self._hidden_objects:
        obj.hide_visual()
    self.scene.update_render(update_sensors=True, ...)

**所以 goal 在 camera1/2/3 里根本不可见**（这是官方设计，不是 bug；实测两个相距 0.3 m 的 goal
在三个相机上都是 0 像素差异）。而且即便可见，`goal_pos` 也在 `obs.extra` 里 —— 模型只要看 state
就能做对，**不需要读指令**。

## 为什么必须是"两个可选物体"

语言条件性的定义是「**指令决定该做什么**」。要让指令**不可被 state 替代**，
场景里就必须存在**多个候选目标**，而"哪个是目标"这一信息**只存在于指令里**。

最小设计：场景内有**两个同形状、不同颜色的方块**，两条指令分别指定抓哪一个；
`obs.extra` 里**不包含**"哪个是目标"。

## 怎么做到零额外示教成本

复用现有 PickCube 示教 —— 其动作轨迹总是抓"槽位 A"（`super()._initialize_episode` 随机化的位置）。

| 变体 | 红块 | 蓝块 | 指令 | 示教动作 |
|---|---|---|---|---|
| `red`  | **A** | B | `pick up the red cube ...` | 抓 A ✓ |
| `blue` | B | **A** | `pick up the blue cube ...` | 抓 A ✓ |

两者的**动作序列完全相同**（都是抓 A、放到 goal），但**颜色位置与指令都不同**。
模型必须"看颜色 + 读指令"才能选对；**忽略指令只能靠猜（约 50%）**。

这不是"为了凑一个红蓝方块的例子"——它是从"指令不可被 state 替代"这个要求**推导出来的必要条件**。
"""

from __future__ import annotations

import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

RED = [1, 0, 0, 1]
BLUE = [0, 0.2, 1, 1]

#: 指令模板（与 DatasetPipe 的 instruction 字段一致，末尾换行由策略适配器补）
INSTRUCTION_RED = "pick up the red cube and move it to the goal position"
INSTRUCTION_BLUE = "pick up the blue cube and move it to the goal position"


@register_env("PickDualCube-v1", max_episode_steps=150)
class PickDualCubeEnv(PickCubeEnv):
    """两个同形状方块（红/蓝）；``target_color`` 决定哪一块是"应被抓的目标"。

    - ``target_color="red"``：红块在**槽位 A**（即示教轨迹抓的位置），蓝块在槽位 B
    - ``target_color="blue"``：蓝块在槽位 A，红块在槽位 B

    无论哪种，**示教动作轨迹（抓 A → 放到 goal）都成立**，因此两者可以共用同一批示教数据，
    只是标签（语言指令）不同。
    """

    def __init__(self, *args, target_color: str = "red", slot_b_offset=(0.0, -0.18), **kwargs):
        if target_color not in ("red", "blue"):
            raise ValueError(f"target_color 必须是 'red' 或 'blue'，收到 {target_color!r}")
        self.target_color = target_color
        self.slot_b_offset = tuple(slot_b_offset)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ 场景
    def _load_scene(self, options: dict):
        # 父类建了 cube(红) + goal_site(hidden)
        super()._load_scene(options)
        # 第二个方块（蓝）——注意它**不在** _hidden_objects 里，因此对相机可见
        self.cube_b = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=BLUE,
            name="cube_b",
            initial_pose=sapien.Pose(p=[0, -0.18, self.cube_half_size]),
        )
        # 父类建 cube 时已固定为红色（color=[1,0,0,1]），无需再设。
        # "哪个是目标"由 _initialize_episode 的**位置交换**体现，而不是改颜色
        # （实测 Actor 上没有 set_color API）。

    # ---------------------------------------------------------------- 初始化
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)  # 随机化 槽位A 与 goal
        with torch.device(self.device):
            b = len(env_idx)
            pos_a = self.cube.pose.p.clone()  # (b,3) 槽位 A = 示教轨迹抓的位置

            # 槽位 B：在 xy 上偏移，z 与 A 相同（都在桌面上）
            off = torch.tensor(self.slot_b_offset, dtype=pos_a.dtype, device=pos_a.device)
            pos_b = pos_a.clone()
            pos_b[:, :2] = pos_b[:, :2] + off

            if self.target_color == "red":
                # 红块留在 A（父类已设），蓝块去 B
                self.cube_b.set_pose(Pose.create_from_pq(pos_b))
            else:
                # 交换：红块去 B，蓝块来 A —— 这样"目标块"仍在槽位 A，示教轨迹依然成立
                self.cube_b.set_pose(Pose.create_from_pq(pos_a))
                self.cube.set_pose(Pose.create_from_pq(pos_b))

    # -------------------------------------------------------------------- obs
    @property
    def _target_cube(self):
        """当前应当被抓取的方块（由指令决定，而不是由位置决定）。"""
        return self.cube if self.target_color == "red" else self.cube_b

    @property
    def _other_cube(self):
        return self.cube_b if self.target_color == "red" else self.cube

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        # 注意：**不**加入 target_color / 哪块是目标 —— 那正是语言要提供的信息
        obs.update(
            cube_b_pose=self.cube_b.pose.raw_pose,
            tcp_to_obj_b_pos=self.cube_b.pose.p - self.agent.tcp_pose.p,
        )
        return obs

    # ---------------------------------------------------------------- 判定
    def evaluate(self):
        target = self._target_cube
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - target.pose.p, axis=1) <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(target)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            # 便于归因：错误方块是否被误送到 goal
            "is_wrong_obj_placed": (
                torch.linalg.norm(self.goal_site.pose.p - self._other_cube.pose.p, axis=1)
                <= self.goal_thresh
            ),
        }

    def instruction(self) -> str:
        """本变体对应的语言指令。"""
        return INSTRUCTION_RED if self.target_color == "red" else INSTRUCTION_BLUE
