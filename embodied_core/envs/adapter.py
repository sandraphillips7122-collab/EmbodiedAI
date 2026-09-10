"""ManiSkill 环境封装（EmbodiedCore · envs）。

本模块把「本机实测得出的 Windows 必需配置」固化下来，避免每个脚本重复踩坑
（完整证据见 ``docs/env_decision.md`` §8）：

1. ``render_backend="cpu"`` —— ManiSkill 默认 ``"gpu"``（``sapien_cuda``）在本机会让
   进程**原生崩溃**（exit 5、无 Python traceback）。改成 ``"cpu"``（``sapien_cpu``）后
   实测可正常渲染与构建 env。
2. ``sim_backend="physx_cpu"`` —— Windows 不支持 GPU 并行仿真。
3. **控制模式只能用关节空间**：Windows 上 SAPIEN 没有 pinocchio（``PinocchioModel = None``），
   凡需要 IK 的 ``pd_ee_*`` 控制模式都会在构建时报
   ``TypeError: 'NoneType' object is not callable``。

评测规范遵循 ManiSkill 官方文档（``user_guide/learning_from_demos/setup.md``）：
``ignore_terminations=True`` + ``reconfiguration_freq=1``，并用 ``CPUGymWrapper`` 记录指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

DEFAULT_TASK = "PickCube-v1"
DEFAULT_SIM_BACKEND = "physx_cpu"
DEFAULT_RENDER_BACKEND = "cpu"
DEFAULT_CONTROL_MODE = "pd_joint_delta_pos"
DEFAULT_OBS_MODE = "rgb"

#: 这些控制模式需要 IK → 依赖 pinocchio → Windows 不可用（见模块 docstring）
IK_REQUIRING_CONTROL_PREFIXES = ("pd_ee", "pd_ee_delta", "pd_ee_pose")


@dataclass
class EnvSpec:
    """一个 ManiSkill 任务的可复现配置。"""

    task: str = DEFAULT_TASK
    obs_mode: str = DEFAULT_OBS_MODE
    control_mode: str = DEFAULT_CONTROL_MODE
    sim_backend: str = DEFAULT_SIM_BACKEND
    render_backend: str = DEFAULT_RENDER_BACKEND
    max_episode_steps: int | None = None
    reconfiguration_freq: int = 1
    #: ManiSkill 的 ``render_mode``。录制 MP4（官方 ``RecordEpisode``）时必须为 ``"rgb_array"``；
    #: 默认 ``None``（不引入额外渲染开销），由 ``RolloutRunner`` 在录制时自动开启。
    render_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for prefix in IK_REQUIRING_CONTROL_PREFIXES:
            if self.control_mode.startswith(prefix):
                raise ValueError(
                    f"control_mode={self.control_mode!r} 需要 IK（pinocchio），本机（Windows）"
                    "没有该依赖，无法构建。请改用关节空间控制，例如 "
                    f"{DEFAULT_CONTROL_MODE!r}。见 docs/env_decision.md §8 约束 2。"
                )
        if self.sim_backend not in {"physx_cpu", "cpu"}:
            raise ValueError(
                f"sim_backend={self.sim_backend!r} 在本机不可用（Windows 无 GPU 仿真）。"
            )
        if self.render_backend in {"gpu", "cuda", "sapien_cuda"}:
            raise ValueError(
                f"render_backend={self.render_backend!r} 在本机会导致进程原生崩溃，"
                "请使用 'cpu'。见 docs/env_decision.md §8 约束 1。"
            )

    def env_kwargs(self) -> dict[str, Any]:
        self.validate()
        kwargs: dict[str, Any] = dict(
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            sim_backend=self.sim_backend,
            render_backend=self.render_backend,
            reconfiguration_freq=self.reconfiguration_freq,
        )
        if self.max_episode_steps is not None:
            kwargs["max_episode_steps"] = self.max_episode_steps
        if self.render_mode is not None:
            kwargs["render_mode"] = self.render_mode
        kwargs.update(self.extra)
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return dict(
            task=self.task,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            sim_backend=self.sim_backend,
            render_backend=self.render_backend,
            max_episode_steps=self.max_episode_steps,
            reconfiguration_freq=self.reconfiguration_freq,
            render_mode=self.render_mode,
            extra=self.extra,
        )


def make_env(spec: EnvSpec, *, record_metrics: bool = True, seed: int | None = None):
    """构建**单环境**（``num_envs=1``）的 CPU 评测环境。

    单环境 + 顺序 rollout 是本项目在 Windows 上最稳的路径：实测 CPU 渲染
    （``render_backend='cpu'``）吞吐 86–200 step/s，100 条 rollout 只需分钟级，
    远低于多进程 spawn 带来的复杂度与风险。
    """
    import mani_skill  # noqa: F401  导入以触发任务注册
    from mani_skill.utils.wrappers import CPUGymWrapper

    spec.validate()
    kwargs = spec.env_kwargs()
    kwargs.setdefault("num_envs", 1)
    env = gym.make(spec.task, **kwargs)
    wrapper = CPUGymWrapper(env, ignore_terminations=True, record_metrics=record_metrics)
    if seed is not None:
        try:
            wrapper.reset(seed=seed)
        except Exception:  # pragma: no cover - 环境若不支持预置 seed 则忽略
            pass
    return wrapper


__all__ = [
    "DEFAULT_TASK",
    "DEFAULT_SIM_BACKEND",
    "DEFAULT_RENDER_BACKEND",
    "DEFAULT_CONTROL_MODE",
    "EnvSpec",
    "make_env",
]
