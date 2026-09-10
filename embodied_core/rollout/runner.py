"""rollout 评测器（EmbodiedCore · rollout）。

职责（项目 A 与项目 B 共用）：

- 用**固定 seed 集**跑确定数量的 episode，输出「成功率 ± 标准误 + Wilson 95% 区间」
- 每个 episode 落一行 JSONL（可追溯到实验记录，见 README 的诚实性约定）
- 标注失败模式（``success`` / ``timeout`` / ``unknown``；Phase 2 起细分）

设计取舍：**顺序单环境**。实测 CPU 渲染（``render_backend='cpu'``）吞吐 86–200 step/s，
100 条 × 150 步 ≈ 2–3 分钟，完全够用；而 Windows 下多进程 spawn 的复杂度与风险更高。

用法::

    python -m embodied_core.rollout.runner --config configs/pickcube_smoke.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
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

    @property
    def total_episodes(self) -> int:
        return len(self.seeds) * self.episodes_per_seed


def _episode_seed(seed: int, episode_idx: int) -> int:
    """把 (seed 块, 块内序号) 映射成互不重复的确定性随机种子。"""
    return seed * 100_000 + episode_idx


def run_rollout(cfg: RolloutConfig, policy: Policy | None = None) -> dict[str, Any]:
    """执行一轮评测并把结果写入 ``experiments/<exp_id>/``。"""
    spec = cfg.env
    env = make_env(spec)
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
                "seeds": list(cfg.seeds),
                "episodes_per_seed": cfg.episodes_per_seed,
                "max_episode_steps": cfg.max_episode_steps,
                "policy": getattr(policy, "name", repr(policy)),
                "action_dim": action_dim,
            },
        )

        episode_idx = 0
        for seed in cfg.seeds:
            for i in range(cfg.episodes_per_seed):
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
    return RolloutConfig(
        exp_id=raw["exp_id"],
        env=env_spec,
        seeds=tuple(rollout_raw.get("seeds", (0, 1, 2))),
        episodes_per_seed=int(rollout_raw.get("episodes_per_seed", 10)),
        max_episode_steps=int(rollout_raw.get("max_episode_steps", 150)),
        policy_name=raw.get("policy", {}).get("name", "zero"),
        experiments_root=raw.get("experiments_root", "experiments"),
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
          f"episodes={cfg.total_episodes} policy={cfg.policy_name}")
    summary = run_rollout(cfg)
    print("[rollout] success_rate: "
          f"{summary['num_successes']}/{summary['num_episodes']} = "
          f"{summary['success_rate']:.3f} "
          f"(±{summary['success_rate_se']:.3f} SE, 95% CI {summary['success_rate_ci95']})")
    print(f"[rollout] failure_modes: {summary['failure_modes']}")
    print(f"[rollout] summary -> experiments/{cfg.exp_id}/summary.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
