"""把两套"目标颜色不同"的 replay 合成**一个**带语言指令差异的 LeRobotDataset。

**为什么必须合并成一个数据集**

如果红/蓝各自一个数据集、分别训练，模型根本不需要读指令 —— 只要认出"我现在是在哪套数据里"
就够了（那是任务泄漏，不是语言条件性）。**只有在同一个数据集里交错出现**，
指令才是**唯一**能区分"该抓哪一块"的信息。

因此本脚本按 ``red_0, blue_0, red_1, blue_1, ...`` **交替**写入，
保证两类样本在数据集里的顺序分布均匀。

**指令**

- red 变体（红块在槽位 A）：``pick up the red cube and move it to the goal position``
- blue 变体（蓝块在槽位 A）：``pick up the blue cube and move it to the goal position``

两者的**动作完全相同**（都抓槽位 A），**只有颜色布局与指令不同**。

用法::

    .venv\\Scripts\\python.exe scripts\\make_dual_dataset.py --per-color 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from embodied_core.data.pipeline import CAMERAS, build_features, build_state
from embodied_core.envs.dual_cube import INSTRUCTION_BLUE, INSTRUCTION_RED

DEMO_DIR = Path.home() / ".maniskill/demos/PickCube-v1/motionplanning"
H5_RED = DEMO_DIR / "trajectory.dualred.rgb.pd_joint_delta_pos.physx_cpu.h5"
H5_BLUE = DEMO_DIR / "trajectory.dualblue.rgb.pd_joint_delta_pos.physx_cpu.h5"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合并 red/blue 两套 replay 为一个带指令的数据集")
    parser.add_argument("--repo-id", default="local/pickcube_dualcolor_200")
    parser.add_argument("--per-color", type=int, default=100, help="每种颜色取多少条 episode")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-videos", action="store_true", default=True)
    args = parser.parse_args(argv)

    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    jobs = [("red", H5_RED, INSTRUCTION_RED), ("blue", H5_BLUE, INSTRUCTION_BLUE)]
    for tag, path, _ in jobs:
        if not path.is_file():
            print(f"[error] 找不到 {tag} 的 replay 产物：{path}")
            print("        先运行 scripts/replay_dual_cube.py --count 200")
            return 2

    # 用 red 的第一条轨迹推断 features（两套的 state/action 维度一致）
    with h5py.File(H5_RED, "r") as fh:
        traj_names = sorted(fh.keys(), key=lambda s: int(s.split("_")[-1]))
        first = fh[traj_names[0]]
        state_dim = build_state(first["obs"]).shape[-1]
        action_dim = int(np.asarray(first["actions"]).shape[-1])

    features = build_features(state_dim, action_dim, use_videos=False)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=str(args.root) if args.root else None,
        robot_type="panda",
        use_videos=False,
    )

    # 预读两套的轨迹名
    traj_lists: dict[str, list[str]] = {}
    for tag, path, _ in jobs:
        with h5py.File(path, "r") as fh:
            names = sorted(fh.keys(), key=lambda s: int(s.split("_")[-1]))
        traj_lists[tag] = names[: args.per_color]
        print(f"[read] {tag}: 使用 {len(traj_lists[tag])} 条 episode")

    n_ep = 0
    # 交替写入：red_0, blue_0, red_1, blue_1, ...
    for i in range(args.per_color):
        for tag, path, instruction in jobs:
            if i >= len(traj_lists[tag]):
                continue
            name = traj_lists[tag][i]
            with h5py.File(path, "r") as fh:
                grp = fh[name]
                actions = np.asarray(grp["actions"], dtype=np.float32)
                states = build_state(grp["obs"])
                images = {
                    cam: np.asarray(grp["obs"]["sensor_data"][cam]["rgb"], dtype=np.uint8)
                    for cam in CAMERAS
                }
            for t in range(len(actions)):
                dataset.add_frame(
                    {
                        "observation.images.camera1": images["camera1"][t],
                        "observation.images.camera2": images["camera2"][t],
                        "observation.images.camera3": images["camera3"][t],
                        "observation.state": states[t],
                        "action": actions[t],
                        "task": instruction,
                    }
                )
            dataset.save_episode()
            n_ep += 1
            if n_ep % 20 == 0:
                print(f"[progress] {n_ep} episodes written", flush=True)

    dataset.finalize()
    print(f"\n[DualDataset] 完成：{n_ep} episodes（red+blue 交替） -> repo_id={args.repo_id}")
    print(f"[DualDataset] state_dim={state_dim} action_dim={action_dim} fps={args.fps}")
    print(f"[DualDataset] 指令：{INSTRUCTION_RED!r} / {INSTRUCTION_BLUE!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
