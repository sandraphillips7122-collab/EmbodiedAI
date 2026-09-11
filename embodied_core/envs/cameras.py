"""项目 B 的 3 相机配置（SmolVLA 需要 camera1 / camera2 / camera3）。

**为什么必须集中一处**：SmolVLA 的 `image_features` 固定为三个相机，而 `PickCube-v1` 默认
只有 `base_camera`。**数据生成（replay）与策略评测（rollout）必须使用完全相同的相机配置**，
否则训练与推理的观测分布不一致，模型表现会莫名其妙地差。

本模块同时被以下两处复用：

- `scripts/replay_3cam.py`（数据生成）
- `embodied_core.envs.adapter.make_env(cameras="three")`（评测/演示时构建 env）
"""

from __future__ import annotations

import numpy as np

#: 相机看向的目标点（与官方 PickCube 的 sensor cam target 一致）
TARGET = [-0.1, 0.0, 0.1]

#: 三个互补视角：正视（同官方 base_camera）/ 侧前方 / 俯视
VIEWS: dict[str, list[float]] = {
    "camera1": [0.3, 0.0, 0.6],
    "camera2": [0.3, 0.45, 0.5],
    "camera3": [0.05, 0.0, 0.75],
}

RESOLUTION = 256
FOV = np.pi / 2


def three_camera_configs():
    """构造 3 个 `CameraConfig`（延迟导入，避免在不需要时引入 ManiSkill 依赖）。"""
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils

    return [
        CameraConfig(
            name,
            sapien_utils.look_at(eye=eye, target=TARGET),
            RESOLUTION,
            RESOLUTION,
            FOV,
            0.01,
            100,
        )
        for name, eye in VIEWS.items()
    ]


def patch_pickcube_with_three_cameras() -> None:
    """把 PickCube 的默认相机替换为上述 3 个。

    注意 `_default_sensor_configs` 是 **property**，必须用 `property(...)` 覆盖。
    """
    import mani_skill.envs.tasks.tabletop.pick_cube as pick_cube_module

    pick_cube_module.PickCubeEnv._default_sensor_configs = property(
        lambda self: three_camera_configs()
    )
