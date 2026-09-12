"""DatasetPipe：ManiSkill replay 的 `.h5` → LeRobotDataset（项目 A / B 共享）。

**用途**：项目 B（SmolVLA）需要 LeRobot 标准数据集；同时这本身也是「数据工程」能力的体现
——不只是「用了现成数据集」。

**关键映射**（与 SmolVLA 的 `config` 对齐，见 `docs/s2_smolvla_smoke.md`）：

| LeRobot 特征 | 来源 | 说明 |
|---|---|---|
| `observation.images.camera1/2/3` | `obs.sensor_data.{camera}/rgb` | 3 个视角，256×256；SmolVLA 的 `image_features` 正是这三个 key |
| `observation.state` | `flatten(agent.qpos, agent.qvel, extra.is_grasped, extra.tcp_pose, extra.goal_pos)` | **29 维**（SmolVLA `max_state_dim=32` 会自行 padding） |
| `action` | `actions` | **8 维**（SmolVLA `max_action_dim=32`） |
| `task` | 调用方传入的指令 | 「语言条件」的关键字段 |

用法::

    python -m embodied_core.data.pipeline --h5 <...trajectory.rgb...h5> \\
        --repo-id local/pickcube_3cam --instruction "pick up the red cube and move it to the goal"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

#: 与 PickCube 的 state 组成一致（ActPolicy 也用同样的展平顺序）
STATE_PARTS = ("qpos", "qvel", "is_grasped", "tcp_pose", "goal_pos")
CAMERAS = ("camera1", "camera2", "camera3")
DEFAULT_INSTRUCTION = "pick up the red cube and move it to the goal position"
DEFAULT_FPS = 20


def build_state(obs_group: h5py.Group) -> np.ndarray:
    """把一条轨迹的观测展平成 (T, 29) 的 state（顺序：qpos, qvel, is_grasped, tcp_pose, goal_pos）。"""
    agent = obs_group["agent"]
    extra = obs_group["extra"]
    parts: list[np.ndarray] = [
        np.asarray(agent["qpos"], dtype=np.float32),
        np.asarray(agent["qvel"], dtype=np.float32),
        np.asarray(extra["is_grasped"], dtype=np.float32).reshape(-1, 1),
        np.asarray(extra["tcp_pose"], dtype=np.float32),
        np.asarray(extra["goal_pos"], dtype=np.float32),
    ]
    return np.concatenate(parts, axis=1)


def build_features(
    state_dim: int, action_dim: int, use_videos: bool = True
) -> dict[str, dict[str, Any]]:
    """构造 LeRobotDataset 的 features 定义。

    注意 ``dtype`` 必须与 ``LeRobotDataset.create(use_videos=...)`` 一致，
    否则 lerobot 会直接报错（``use_videos is set to False`` 时不能声明 ``dtype="video"``）。
    """
    image_dtype = "video" if use_videos else "image"
    features: dict[str, dict[str, Any]] = {}
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": image_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {"dtype": "float32", "shape": (state_dim,), "names": None}
    features["action"] = {"dtype": "float32", "shape": (action_dim,), "names": None}
    return features


def convert(
    h5_path: str | Path,
    repo_id: str,
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    fps: int = DEFAULT_FPS,
    max_episodes: int | None = None,
    root: str | Path | None = None,
    use_videos: bool = True,
) -> Path:
    """把 ManiSkill replay 的 `.h5` 转成 LeRobotDataset，返回落盘目录。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as fh:
        traj_names = sorted(fh.keys(), key=lambda s: int(s.split("_")[-1]))
        if max_episodes is not None:
            traj_names = traj_names[:max_episodes]

        first = fh[traj_names[0]]
        state0 = build_state(first["obs"])
        action_dim = int(np.asarray(first["actions"]).shape[-1])
        features = build_features(state0.shape[-1], action_dim, use_videos)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=str(root) if root else None,
            robot_type="panda",
            use_videos=use_videos,
        )

        for name in traj_names:
            grp = fh[name]
            actions = np.asarray(grp["actions"], dtype=np.float32)
            states = build_state(grp["obs"])
            images = {
                cam: np.asarray(grp["obs"]["sensor_data"][cam]["rgb"], dtype=np.uint8)
                for cam in CAMERAS
            }
            # 观测比动作多 1 帧（首帧为初始观测），对齐到动作长度
            horizon = len(actions)
            for t in range(horizon):
                frame = {
                    "observation.images.camera1": images["camera1"][t],
                    "observation.images.camera2": images["camera2"][t],
                    "observation.images.camera3": images["camera3"][t],
                    "observation.state": states[t],
                    "action": actions[t],
                    "task": instruction,
                }
                dataset.add_frame(frame)
            dataset.save_episode()

        dataset.finalize()

    out = Path(root) if root else None
    print(f"[DatasetPipe] 转换完成：{len(traj_names)} episodes -> repo_id={repo_id}")
    print(f"[DatasetPipe] state_dim={state0.shape[-1]} action_dim={action_dim} fps={fps}")
    return out if out else Path(repo_id)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ManiSkill .h5 → LeRobotDataset")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-videos", action="store_true", help="存成图片而不是 MP4")
    args = parser.parse_args(list(argv) if argv is not None else None)

    convert(
        args.h5,
        args.repo_id,
        instruction=args.instruction,
        fps=args.fps,
        max_episodes=args.max_episodes,
        root=args.root,
        use_videos=not args.no_videos,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
