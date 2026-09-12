"""用 ``PickDualCube-v1`` 重放现有示教，生成两套"目标颜色不同"的 3 相机数据。

**为什么可以零额外示教成本**

现有 PickCube 示教的动作轨迹总是"抓槽位 A、放到 goal"。在 ``PickDualCube-v1`` 里：
- ``target_color="red"`` ：红块在 A、蓝块在 B → 抓 A 的动作**正好抓的是红块**
- ``target_color="blue"``：蓝块在 A、红块在 B → 同一批动作**正好抓的是蓝块**

因此**同一批示教可以产出两条指令的样本**，不需要重新做 motion planning。

**相机**：复用 ``scripts/replay_3cam.py`` 的 patch（覆盖 ``PickCubeEnv._default_sensor_configs``），
``PickDualCubeEnv`` 继承自 ``PickCubeEnv``，所以 patch 自动生效。

**产物**：每个 target_color 一个 h5，含 3 个相机的 rgb。

用法::

    .venv\\Scripts\\python.exe scripts\\replay_dual_cube.py --count 200
    .venv\\Scripts\\python.exe scripts\\replay_dual_cube.py --count 200 --colors red blue
"""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

import mani_skill.envs.tasks.tabletop.pick_cube as pick_cube_module

# 关键：import 以触发 @register_env —— 否则 gym.make("PickDualCube-v1") 会报"未注册"
import embodied_core.envs.dual_cube  # noqa: F401

DEFAULT_DEMO_DIR = (
    Path.home() / ".maniskill/demos/PickCube-v1/motionplanning"
)
TARGET = [-0.1, 0.0, 0.1]
VIEWS = {
    "camera1": [0.3, 0.0, 0.6],
    "camera2": [0.3, 0.45, 0.5],
    "camera3": [0.05, 0.0, 0.75],
}
RESOLUTION = 256
FOV = np.pi / 2


def _three_camera_configs(self):
    """与 ``scripts/replay_3cam.py`` 完全一致的 3 相机配置（保证训练/评测观测一致）。"""
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


def _b_slot_row(z: float = 0.02) -> list[float]:
    """槽位 B 的固定 pose（pos 3 + quat 4 + 线速度 3 + 角速度 3 = 13 维）。"""
    return [0.0, -0.20, z, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _make_dual_h5(src_h5: Path, dst_h5: Path, target_color: str) -> None:
    """复制示教 h5，并在每条轨迹的 ``env_states/actors`` 里补上 ``cube_b``。

    **为什么必须在 h5 层做**：``replay_trajectory`` 用示教记录的 ``env_states`` 递归覆盖
    新环境状态（``replay_trajectory.py:199`` 的 ``recursive_replace``），而 ``env_states``
    **存在 h5 里、不在 trajectory.json 里**（json 的 ``episodes[i]`` 只有
    ``episode_id`` / ``episode_seed`` / ``control_mode`` / ``elapsed_steps`` /
    ``reset_kwargs`` / ``success`` 这几个键）。

    新环境多了一个 ``cube_b`` actor，示教里没有这个键 → ``recursive_replace`` 直接
    ``KeyError: 'cube_b'``。所以要把这个键补进每条轨迹。

    ``target_color`` 决定红/蓝谁在"槽位 A"（示教轨迹抓的位置）：
    - ``red`` ：红块留在示教位置，蓝块放固定的槽位 B
    - ``blue``：蓝块占据示教位置（于是同一批动作抓的就是蓝块），红块挪到槽位 B
    """
    import h5py

    with h5py.File(src_h5, "r") as fi, h5py.File(dst_h5, "w") as fo:
        # 整份复制，之后再改 env_states
        for key in fi.keys():
            fi.copy(key, fo, name=key)

        n_patched = 0
        for traj in fi.keys():
            if "env_states/actors/cube" not in fi[traj]:
                continue
            cube = np.asarray(fi[f"{traj}/env_states/actors/cube"], dtype=np.float32)  # (T,13)
            t = cube.shape[0]
            fixed = np.repeat(np.asarray([_b_slot_row()], dtype=np.float32), t, axis=0)

            actors = fo[f"{traj}/env_states/actors"]
            if "cube_b" in actors:
                del actors["cube_b"]
            if target_color == "red":
                actors.create_dataset("cube_b", data=fixed)
            else:
                actors.create_dataset("cube_b", data=cube)
                del fo[f"{traj}/env_states/actors/cube"]
                fo[f"{traj}/env_states/actors"].create_dataset("cube", data=fixed)
            n_patched += 1

    print(f"[patch] target_color={target_color}: 已为 {n_patched} 条轨迹注入 cube_b  -> {dst_h5.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用 PickDualCube-v1 重放示教（两个目标颜色）")
    parser.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR))
    parser.add_argument("--count", type=int, default=200, help="replay 的 episode 数")
    parser.add_argument("--colors", nargs="+", default=["red", "blue"])
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--sim-backend", default="physx_cpu")
    args = parser.parse_args(argv)

    demo_dir = Path(args.demo_dir)
    traj_h5 = demo_dir / "trajectory.h5"
    traj_json = demo_dir / "trajectory.json"
    if not traj_h5.is_file() or not traj_json.is_file():
        print(f"[error] 找不到示教文件：{traj_h5} / {traj_json}")
        return 2

    # 备份原始 json（只做一次）
    backup = demo_dir / "trajectory.json.dualcube_backup"
    if not backup.exists():
        shutil.copy(traj_json, backup)
        print(f"[backup] {traj_json} -> {backup}")

    original = json.loads(backup.read_text(encoding="utf-8"))

    # 自检：确认 env_states 确实存在 h5 里（否则下面的注入会静默跳过，白跑一遍 replay）
    import h5py

    with h5py.File(traj_h5, "r") as f:
        first = sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))[0]
        if "env_states/actors/cube" not in f[first]:
            print(f"[error] {first} 里没有 env_states/actors/cube，无法注入 cube_b")
            return 2
        print(f"[check] {first}/env_states/actors = {sorted(f[first]['env_states/actors'].keys())}")

    # 相机 patch（PickDualCubeEnv 继承 PickCubeEnv，故自动生效）
    pick_cube_module.PickCubeEnv._default_sensor_configs = property(_three_camera_configs)
    print(f"[replay_dual_cube] cameras={list(VIEWS)} res={RESOLUTION}")

    produced: list[Path] = []
    for color in args.colors:
        cfg = json.loads(json.dumps(original))  # 深拷贝
        env_info = cfg.setdefault("env_info", {})
        # 关键：replay_trajectory 用 env_info["env_id"] 来 gym.make（见 replay_trajectory.py:485），
        # 不是 env_class —— 改错字段会继续用 PickCube-v1 建环境，随后因 target_color 报 TypeError。
        env_info["env_id"] = "PickDualCube-v1"
        env_info["env_class"] = "PickDualCube-v1"
        # 我们的示教平均 77 步，默认 50 会被截断（项目 A 踩过同一个坑）
        env_info["max_episode_steps"] = 150
        kwargs = env_info.setdefault("env_kwargs", {})
        kwargs["target_color"] = color
        kwargs["max_episode_steps"] = 150
        kwargs["render_backend"] = "cpu"  # Windows 硬约束

        # 生成"补过 cube_b"的 h5（同名 json 会被 replay_trajectory 自动识别）
        dual_h5 = demo_dir / f"trajectory.dual{color}.h5"
        dual_json = demo_dir / f"trajectory.dual{color}.json"
        _make_dual_h5(traj_h5, dual_h5, color)
        dual_json.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"\n[replay] target_color={color}  env_id=PickDualCube-v1  count={args.count}")

        sys.argv = [
            "replay_trajectory",
            "--traj-path",
            str(dual_h5),
            "--use-first-env-state",
            "-c",
            args.control_mode,
            "-o",
            "rgb",
            "--save-traj",
            "-b",
            args.sim_backend,
            "--num-envs",
            "1",
            "--count",
            str(args.count),
        ]
        _replay_prod = demo_dir / f"trajectory.rgb.{args.control_mode}.{args.sim_backend}.h5"
        mtime_before = _replay_prod.stat().st_mtime if _replay_prod.is_file() else 0.0

        runpy.run_module("mani_skill.trajectory.replay_trajectory", run_name="__main__")

        # 官方产物名固定 -> 改名以免被下一轮覆盖。
        # ⚠️ 曾经因为**无条件** rename 而误伤同目录下的既有文件：
        #    它把 PickCube 的 `trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5`
        #    （3 相机 1000 条，5.05 GB）rename 成了 dual 产物，第二次 replay 又覆盖了它。
        #    因此这里加两道保护：
        #      1) 只有本次 replay 真的改写了该文件（mtime 变化）才重命名
        #      2) 目标已存在时**报错**而不是覆盖
        default_out = demo_dir / f"trajectory.rgb.{args.control_mode}.{args.sim_backend}.h5"
        target_out = demo_dir / f"trajectory.dual{color}.rgb.{args.control_mode}.{args.sim_backend}.h5"
        if not default_out.is_file():
            print(f"[warn] 未找到官方产物 {default_out}")
        elif default_out.stat().st_mtime <= mtime_before:
            print(f"[warn] {default_out.name} 未被本次 replay 改写，跳过重命名（避免误伤既有数据）")
        elif target_out.exists():
            raise SystemExit(
                f"[error] 目标已存在，拒绝覆盖：{target_out}\n"
                f"        如确需重跑，请先手动移走该文件。"
            )
        else:
            default_out.rename(target_out)
            produced.append(target_out)
            print(f"[rename] {default_out.name} -> {target_out.name}")

    # 恢复原始 json，避免污染后续（如原 PickCube 的 3 相机 replay）
    traj_json.write_text(json.dumps(original, indent=2), encoding="utf-8")
    print(f"\n[restore] {traj_json} 已恢复为原始内容")

    print("\n=== 产物 ===")
    for p in produced:
        print(f"  {p}  ({p.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
