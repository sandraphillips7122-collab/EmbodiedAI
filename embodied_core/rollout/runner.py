"""rollout 评测器（EmbodiedCore · rollout）。

职责（项目 A 与项目 B 共用）：

- 用**固定 seed 集**跑确定数量的 episode，输出「成功率 ± 标准误 + Wilson 95% 区间」
- 每个 episode 落一行 JSONL（可追溯到实验记录，见 README 的诚实性约定）
- 标注失败模式（``success`` / ``timeout`` / ``unknown``；更细分类见后续工作）
- 可选录制 MP4（只录前 ``num_videos`` 个 episode）

设计取舍：**顺序单环境**。实测 CPU 渲染吞吐 86–200 step/s，100 条 rollout 只需分钟级。

## 针对 Windows TDR 的保护（重要）

本机（Windows + RTX 4060）上，**长时间连续「3 相机渲染 + VLA 推理」会触发驱动 TDR**
（事件日志 `nvlddmkm` ID 153 = 显示驱动停止响应并已恢复），表现为
``RuntimeError: vk::Device::waitSemaphores: ErrorDeviceLost``。此时**显存是充足的**
（实测剩余 5.9 GB），所以**不是 OOM**。应对手段：

- ``sync_and_free``：每步 ``torch.cuda.synchronize()``，每个 episode 结束 ``gc.collect()``
  + ``torch.cuda.empty_cache()``
- ``rebuild_env_every``：每 N 个 episode 关闭并重建环境，释放 SAPIEN 的渲染资源

用法::

    python -m embodied_core.rollout.runner --config configs/pickcube_smoke.yaml
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from embodied_core.envs.adapter import EnvSpec, make_env
from embodied_core.metrics.recorder import EpisodeResult, ExperimentRecorder
from embodied_core.rollout.policies import Policy, build_policy


@dataclass
class RolloutConfig:
    """一次评测的配置。"""

    exp_id: str
    env: EnvSpec
    seeds: tuple[int, ...] = (0, 1, 2)
    episodes_per_seed: int = 10
    max_episode_steps: int = 150
    policy_name: str = "zero"
    experiments_root: str = "experiments"
    #: 相机配置：``"default"``（任务自带）或 ``"three"``（项目 B 的 camera1/2/3）
    cameras: str = "default"
    #: 录制 MP4（用 ManiSkill 官方 RecordEpisode，按 episode 为单位输出）
    record_video: bool = False
    #: 只录前多少个 episode（0 表示不录）；控制磁盘占用
    num_videos: int = 2
    video_fps: int = 30
    #: 每步做 CUDA 同步，且每 episode 结束释放显存/内存（缓解 Windows TDR）
    sync_and_free: bool = False
    #: 每多少个 episode 重建一次环境（0 = 不重建）；进一步缓解 TDR
    rebuild_env_every: int = 0

    @property
    def total_episodes(self) -> int:
        return len(self.seeds) * self.episodes_per_seed

    @property
    def video_dir(self) -> Path:
        return Path(self.experiments_root) / self.exp_id / "videos"


def _episode_seed(seed: int, episode_idx: int) -> int:
    """把 (seed 块, 块内序号) 映射成互不重复的确定性随机种子。"""
    return seed * 100_000 + episode_idx


def _sync() -> None:
    """轻量 CUDA 同步（仅在可用时）。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover
        pass


def _free() -> None:
    """释放 Python 内存与 CUDA 缓存（缓解长时间运行的资源累积）。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _wrap_video(env, cfg: RolloutConfig):
    """在 CPUGymWrapper **之外**再包一层官方 ``RecordEpisode``。

    顺序与 ManiSkill 官方 baseline（``act/make_env.py``）一致：内层 env → CPUGymWrapper →
    RecordEpisode。只对前 ``num_videos`` 个 episode 触发录制（``save_video_trigger``）。
    """
    from mani_skill.utils.wrappers import RecordEpisode

    limit = max(0, int(cfg.num_videos))
    if limit == 0:
        return env
    cfg.video_dir.mkdir(parents=True, exist_ok=True)
    return RecordEpisode(
        env,
        output_dir=str(cfg.video_dir),
        save_trajectory=False,
        save_video=True,
        # ManiSkill 的语义：trigger 的参数是「已流逝的环境步数」，不是 episode 序号
        # （见 mani_skill/utils/wrappers/record.py 的 save_video_trigger 文档）。
        # 因此用 num_videos × max_episode_steps 作上限，从而只录前若干个 episode。
        save_video_trigger=lambda elapsed_steps: (
            elapsed_steps < limit * max(1, cfg.max_episode_steps)
        ),
        info_on_video=True,
        video_fps=cfg.video_fps,
        avoid_overwriting_video=True,
        source_type="embodied_core",
        source_desc="rollout evaluation",
    )


def _build_env(cfg: RolloutConfig):
    """按配置构建（可选的）带录制 env，返回 (实际使用的 spec, env)。"""
    spec = cfg.env
    if cfg.record_video and spec.render_mode is None:
        # 官方 RecordEpisode 依赖 env 的 render_mode；仅在录制时开启，避免常规评测多付渲染开销
        spec = replace(spec, render_mode="rgb_array")
    env = make_env(spec, cameras=cfg.cameras)
    if cfg.record_video:
        env = _wrap_video(env, cfg)
    return spec, env


def run_rollout(cfg: RolloutConfig, policy: Policy | None = None) -> dict[str, Any]:
    """执行一轮评测并把结果写入 ``experiments/<exp_id>/``。"""
    spec, env = _build_env(cfg)
    try:
        action_space = env.action_space
        action_dim = int(np.prod(action_space.shape))
        if policy is None:
            policy = build_policy(cfg.policy_name, action_space=action_space, action_dim=action_dim)

        recorder = ExperimentRecorder(
            cfg.experiments_root,
            cfg.exp_id,
            meta={
                "env": spec.to_dict(),
                "cameras": cfg.cameras,
                "seeds": list(cfg.seeds),
                "episodes_per_seed": cfg.episodes_per_seed,
                "max_episode_steps": cfg.max_episode_steps,
                "policy": getattr(policy, "name", repr(policy)),
                "action_dim": action_dim,
                "record_video": cfg.record_video,
                "num_videos": cfg.num_videos if cfg.record_video else 0,
                "video_fps": cfg.video_fps,
                "sync_and_free": cfg.sync_and_free,
                "rebuild_env_every": cfg.rebuild_env_every,
            },
        )

        episode_idx = 0
        for seed in cfg.seeds:
            for i in range(cfg.episodes_per_seed):
                # 定期重建环境：释放 SAPIEN 渲染资源，规避 Windows TDR
                if (
                    cfg.rebuild_env_every > 0
                    and episode_idx > 0
                    and episode_idx % cfg.rebuild_env_every == 0
                ):
                    env.close()
                    _free()
                    spec, env = _build_env(cfg)

                ep_seed = _episode_seed(seed, i)
                obs, _info = env.reset(seed=ep_seed)
                policy.reset(seed=ep_seed)

                started = time.perf_counter()
                success = False
                steps = 0
                terminated = truncated = False
                info: dict[str, Any] = {}
                while not (terminated or truncated) and steps < cfg.max_episode_steps:
                    action = policy.predict(obs)
                    obs, _reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    if cfg.sync_and_free:
                        _sync()
                    if bool(info.get("success", False)):
                        success = True
                duration = time.perf_counter() - started

                if success:
                    failure = "success"
                elif truncated or steps >= cfg.max_episode_steps:
                    failure = "timeout"
                else:
                    failure = "unknown"

                recorder.log_episode(
                    EpisodeResult(
                        exp_id=cfg.exp_id,
                        task=spec.task,
                        policy=getattr(policy, "name", repr(policy)),
                        seed=seed,
                        episode_idx=episode_idx,
                        success=success,
                        steps=steps,
                        duration_s=duration,
                        failure_mode=failure,
                    )
                )
                episode_idx += 1

                if cfg.sync_and_free:
                    _free()

        summary = recorder.write_summary()
        summary["episodes_per_seed"] = cfg.episodes_per_seed
        return summary
    finally:
        env.close()


def config_from_dict(raw: dict[str, Any]) -> RolloutConfig:
    """从（已解析的 YAML）字典构造配置。"""
    env_raw = dict(raw.get("env", {}))
    env_spec = EnvSpec(**env_raw)
    rollout_raw = raw.get("rollout", {})
    video_raw = dict(raw.get("video", {}))
    resources_raw = dict(raw.get("resources", {}))
    return RolloutConfig(
        exp_id=raw["exp_id"],
        env=env_spec,
        seeds=tuple(rollout_raw.get("seeds", (0, 1, 2))),
        episodes_per_seed=int(rollout_raw.get("episodes_per_seed", 10)),
        max_episode_steps=int(rollout_raw.get("max_episode_steps", 150)),
        policy_name=raw.get("policy", {}).get("name", "zero"),
        experiments_root=raw.get("experiments_root", "experiments"),
        cameras=raw.get("cameras", "default"),
        record_video=bool(video_raw.get("record_video", False)),
        num_videos=int(video_raw.get("num_videos", 2)),
        video_fps=int(video_raw.get("fps", 30)),
        sync_and_free=bool(resources_raw.get("sync_and_free", False)),
        rebuild_env_every=int(resources_raw.get("rebuild_env_every", 0)),
    )


def load_config(path: str | Path) -> RolloutConfig:
    import yaml

    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EmbodiedCore rollout 评测器")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    print(f"[rollout] exp_id={cfg.exp_id} task={cfg.env.task} "
          f"episodes={cfg.total_episodes} policy={cfg.policy_name} cameras={cfg.cameras}")
    summary = run_rollout(cfg)
    print("[rollout] success_rate: "
          f"{summary['num_successes']}/{summary['num_episodes']} = "
          f"{summary['success_rate']:.3f} "
          f"(±{summary['success_rate_se']:.3f} SE, 95% CI {summary['success_rate_ci95']})")
    print(f"[rollout] failure_modes: {summary['failure_modes']}")
    print(f"[rollout] summary -> experiments/{cfg.exp_id}/summary.json")
    if cfg.record_video:
        print(f"[rollout] videos -> {cfg.video_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
