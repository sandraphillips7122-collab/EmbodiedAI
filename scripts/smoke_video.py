"""MP4 录制冒烟测试。

验证 ``RolloutRunner`` 能真正产出可播放的 MP4（用官方 ``RecordEpisode``）。
默认写到工作区外的临时目录，避免测试产物污染 ``experiments/``。

用法::

    .venv\\Scripts\\python.exe scripts\\smoke_video.py
    .venv\\Scripts\\python.exe scripts\\smoke_video.py --root experiments  # 写到仓库内
"""

from __future__ import annotations

import argparse
import json

from embodied_core.envs.adapter import EnvSpec
from embodied_core.rollout.runner import RolloutConfig, run_rollout


def main() -> int:
    parser = argparse.ArgumentParser(description="MP4 录制冒烟测试")
    parser.add_argument("--root", default="experiments", help="experiments 根目录")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    cfg = RolloutConfig(
        exp_id="smoke_video",
        env=EnvSpec(),
        seeds=(0,),
        episodes_per_seed=args.episodes,
        max_episode_steps=args.max_steps,
        policy_name="zero",
        experiments_root=args.root,
        record_video=True,
        num_videos=args.episodes,
    )
    summary = run_rollout(cfg)
    print(json.dumps(
        {k: summary[k] for k in ("num_episodes", "num_successes", "failure_modes")},
        ensure_ascii=False,
    ))

    files = sorted(p for p in cfg.video_dir.rglob("*") if p.is_file())
    print(f"[video] {len(files)} file(s) under {cfg.video_dir}")
    for path in files:
        print(f"  {path.name}  {path.stat().st_size} bytes")

    mp4s = [p for p in files if p.suffix.lower() == ".mp4" and p.stat().st_size > 1000]
    print("VIDEO_OK" if mp4s else "VIDEO_MISSING")
    return 0 if mp4s else 1


if __name__ == "__main__":
    raise SystemExit(main())
