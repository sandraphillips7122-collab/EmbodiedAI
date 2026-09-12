"""可行性验证：从 3 个相机能否**分辨不同的 goal 位置**。

**为什么这是前提**

第五步要做"真正语言条件"的任务：不同指令对应不同正确动作（例如
"move the cube to the left target" vs "move the cube to the right target"）。
这要求：**目标位置必须能从图像上被区分** —— 否则模型"看"不出该去哪个，
实验就退化成"模型忽略语言也能蒙对"，无法证明语言条件性。

PickCube 的 `goal_site` 是一个半径 = `goal_thresh`（默认 0.025 m）的绿色球，
默认采样在 x,y ∈ ±`cube_spawn_half_size`（默认 ±0.05，即**只有 10 cm 跨度**）。
这个跨度在当前 3 个相机视角下是否可分辨，必须实测而不是假设。

本脚本：
1. 固定相机与机器人初始状态，只改 goal 位置；
2. 对若干组分离距离（左右）分别截图；
3. 计算像素差异（定量），并保存并排对比图供人工确认。

用法::

    .venv\\Scripts\\python.exe scripts/check_goal_visibility.py --out experiments/B_goal_vis
    .venv\\Scripts\\python.exe scripts/check_goal_visibility.py --dx 0.10 0.30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMERAS = ("camera1", "camera2", "camera3")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="验证 goal 位置是否可从 3 相机分辨")
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments/B_goal_vis"))
    parser.add_argument("--dx", type=float, nargs="+", default=[0.10, 0.20, 0.30],
                        help="左右 goal 的 x 方向分离距离（米）")
    parser.add_argument("--y", type=float, default=0.0, help="goal 的 y 坐标")
    parser.add_argument("--z", type=float, default=0.03, help="goal 的 z 坐标")
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    from embodied_core.envs.adapter import EnvSpec, make_env

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(EnvSpec(max_episode_steps=50), cameras="three")
    try:
        obs, _ = env.reset(seed=0)
        unwrapped = env.unwrapped

        def set_goal(x: float, y: float, z: float) -> dict:
            """把 goal_site 放到指定位置并取一张 3 相机观测。"""
            pose = torch.tensor([[x, y, z]], dtype=torch.float32)
            from mani_skill.utils.structs import Pose

            unwrapped.goal_site.set_pose(Pose.create_from_pq(pose))
            unwrapped.goal_pos = pose
            # 只需要重新取观测（不推进物理），保证两次截图的其他状态完全一致
            new_obs = unwrapped.get_obs(unflattened=True)
            return new_obs

        report: dict = {"dxy": {}, "note": "只改 goal 位置，其余状态固定"}
        for dx in args.dx:
            obs_left = set_goal(-dx / 2, args.y, args.z)
            obs_right = set_goal(+dx / 2, args.y, args.z)

            per_cam: dict = {}
            for cam in CAMERAS:
                img_l = np.asarray(obs_left["sensor_data"][cam]["rgb"])
                img_r = np.asarray(obs_right["sensor_data"][cam]["rgb"])
                if img_l.ndim == 4:
                    img_l, img_r = img_l[0], img_r[0]
                diff = np.abs(img_l.astype(np.int16) - img_r.astype(np.int16))
                changed = int((diff.max(axis=-1) > 8).sum())
                per_cam[cam] = {
                    "changed_pixels": changed,
                    "total_pixels": int(img_l.shape[0] * img_l.shape[1]),
                    "changed_ratio": changed / float(img_l.shape[0] * img_l.shape[1]),
                    "mean_abs_diff": float(diff.mean()),
                }
                # 并排保存（左 | 右 | 差异放大）
                side = np.concatenate([img_l, img_r, np.clip(diff * 4, 0, 255).astype(np.uint8)], axis=1)
                try:
                    from PIL import Image

                    Image.fromarray(side).save(out_dir / f"dx{int(dx*100):03d}_{cam}.png")
                except Exception:
                    # 没有 PIL 就存 npy
                    np.save(out_dir / f"dx{int(dx*100):03d}_{cam}.npy", side)

            report["dxy"][f"dx={dx}"] = per_cam

            print(f"\n=== goal 分离 {dx:+.2f} m（x 方向）===")
            for cam in CAMERAS:
                s = per_cam[cam]
                print(f"  {cam}: 变化像素 {s['changed_pixels']:>6}/{s['total_pixels']} "
                      f"({100*s['changed_ratio']:.3f}%)  平均绝对差 {s['mean_abs_diff']:.3f}")

        (out_dir / "goal_visibility.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[wrote] {out_dir}/goal_visibility.json（并排图也在同目录）")
        print("\n判读建议：看 changed_ratio 是否显著大于 0，并人工确认并排图里 "
              "goal 球的位置差异是否肉眼可见、且不被机器人手臂遮挡。")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
