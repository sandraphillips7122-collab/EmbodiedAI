"""环境相关的统一封装。"""

from embodied_core.envs.adapter import (
    DEFAULT_CONTROL_MODE,
    DEFAULT_RENDER_BACKEND,
    DEFAULT_SIM_BACKEND,
    DEFAULT_TASK,
    EnvSpec,
    make_env,
)

__all__ = [
    "DEFAULT_CONTROL_MODE",
    "DEFAULT_RENDER_BACKEND",
    "DEFAULT_SIM_BACKEND",
    "DEFAULT_TASK",
    "EnvSpec",
    "make_env",
]
