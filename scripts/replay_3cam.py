"""用 3 个相机重新 replay PickCube 示教（供项目 B 的 SmolVLA 使用）。

**为什么需要**：SmolVLA 的 `image_features` 固定为 `camera1` / `camera2` / `camera3`
（各 `(3, 256, 256)`），而 `PickCube-v1` 默认只暴露 1 个 `base_camera`（128×128）。

**做法**：不修改第三方代码，只把任务的 `_default_sensor_configs`（注意是 **property**）
替换成 3 个不同视角的相机，然后交给官方 `replay_trajectory` 执行——这样仍然复用官方的
replay/录观测逻辑，只是相机变了。

用法::

    .venv\\Scripts\\python.exe scripts\\replay_3cam.py ^
        --traj-path "%USERPROFILE%\\.maniskill\\demos\\PickCube-v1\\motionplanning\\trajectory.h5" ^
        --use-first-env-state -c pd_joint_delta_pos -o rgb --save-traj ^
        -b physx_cpu --num-envs 1 --count 100

产物：`trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5`（含 3 个相机的 rgb）
"""

from __future__ import annotations

import runpy
import sys

import numpy as np

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

import mani_skill.envs.tasks.tabletop.pick_cube as pick_cube_module

#: 相机看向的目标点（与官方 PickCube 的 sensor cam target 一致）
TARGET = [-0.1, 0.0, 0.1]

#: 三个互补视角：默认正视 / 侧视 / 俯视
VIEWS: dict[str, list[float]] = {
    "camera1": [0.3, 0.0, 0.6],    # 与官方 base_camera 相同
    "camera2": [0.3, 0.45, 0.5],   # 侧前方
    "camera3": [0.05, 0.0, 0.75],  # 俯视
}

RESOLUTION = 256
FOV = np.pi / 2


def _three_camera_configs(self):
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


def main() -> int:
    # 注意：_default_sensor_configs 是 property，必须用 property(...) 覆盖
    pick_cube_module.PickCubeEnv._default_sensor_configs = property(_three_camera_configs)
    print(f"[replay_3cam] cameras={list(VIEWS)} resolution={RESOLUTION}")

    sys.argv = ["replay_trajectory", *sys.argv[1:]]
    runpy.run_module("mani_skill.trajectory.replay_trajectory", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
