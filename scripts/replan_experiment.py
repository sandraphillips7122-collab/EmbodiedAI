"""`n_action_steps`（重规划频率）控制变量实验 —— 不重训。

**动机**（来自 `docs/latency_report.md` 的实测）：

SmolVLA 用 `chunk_size = n_action_steps = 50`：模型一次预测 50 个动作，然后**开环执行 50 步**。
也就是说，这 50 步里**只有第 1 步是基于"当前"观测算出来的**，其余 49 步都是按**已经过时的观测**
规划的动作。延迟拆分显示每 50 步还有一次 ≈0.43 s 的 forward 尖峰，其间环境继续推进。

**要回答的问题**：当前极低成功率（v2: 1/60 = 0.0167）有多少来自"长时间开环执行"，
而不是模型本身完全没学会？

**控制变量设计**（除 `n_action_steps` 外全部固定）：

| 固定项 | 值 |
|---|---|
| checkpoint | `B_smolvla_finetune_v2/checkpoints/020000`（两版都没有重训） |
| 任务 / 控制模式 | `PickCube-v1` / `pd_joint_delta_pos` |
| 相机 | 3 相机（`cameras="three"`） |
| seeds / 每 seed 集数 | 由命令行给定，**所有配置完全相同** |
| 评测器 | `embodied_core.rollout.runner`（与项目 A/B 同口径） |
| 变化项 | **仅** `policy.config.n_action_steps` |

`chunk_size` **保持不变**（模型仍然一次性预测 50 个动作），只改"实际执行/重规划多少个"，
因此这不是"换模型"，而是纯粹的**闭环执行策略**变量。

用法::

    .venv\\Scripts\\python.exe scripts/replan_experiment.py
    .venv\\Scripts\\python.exe scripts/replan_experiment.py --n-action-steps 5 10 25 50 \
        --seeds 0 1 2 --episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT = REPO_ROOT / "experiments/B_smolvla_finetune_v2/checkpoints/020000/pretrained_model"
INSTRUCTION = "pick up the red cube and move it to the goal position"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="n_action_steps 控制变量实验（不重训）")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25, 50],
        help="要比较的 n_action_steps 取值",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seed 块")
    parser.add_argument("--episodes", type=int, default=5, help="每个 seed 块的 episode 数")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--ckpt", default=str(CKPT))
    parser.add_argument("--out", default=None, help="把汇总写成 json")
    args = parser.parse_args(argv)

    from embodied_core.envs.adapter import EnvSpec
    from embodied_core.rollout.runner import RolloutConfig, run_rollout
    from embodied_core.rollout.smolvla_policy import SmolVLAPolicyAdapter

    print(f"[setup] 加载 SmolVLA（只加载一次，全程复用同一实例）: {args.ckpt}", flush=True)
    adapter = SmolVLAPolicyAdapter(args.ckpt, instruction=INSTRUCTION, device="cuda")
    chunk_size = int(getattr(adapter.policy.config, "chunk_size", -1))
    print(f"[setup] chunk_size = {chunk_size}（保持不变）", flush=True)

    total_eps = len(args.seeds) * args.episodes
    rows = []
    for n in args.n_action_steps:
        # 只改这一个变量
        adapter.policy.config.n_action_steps = int(n)
        adapter.policy.reset()

        effective = int(getattr(adapter.policy.config, "n_action_steps", -1))
        print(
            f"\n[run] n_action_steps={n} (生效值={effective})  "
            f"seeds={args.seeds} x {args.episodes} = {total_eps} episodes",
            flush=True,
        )

        cfg = RolloutConfig(
            exp_id=f"B_replan_n{n}",
            env=EnvSpec(max_episode_steps=args.max_steps),
            seeds=tuple(args.seeds),
            episodes_per_seed=args.episodes,
            max_episode_steps=args.max_steps,
            policy_name="smolvla",
            cameras="three",
            sync_and_free=True,
            rebuild_env_every=4,
            record_video=False,
        )
        summary = run_rollout(cfg, policy=adapter)
        row = {
            "n_action_steps": n,
            "n_action_steps_effective": effective,
            "num_episodes": summary["num_episodes"],
            "num_successes": summary["num_successes"],
            "success_rate": summary["success_rate"],
            "success_rate_se": summary["success_rate_se"],
            "success_rate_ci95": summary["success_rate_ci95"],
            "duration_s_mean": summary["duration_s_mean"],
            "failure_modes": summary["failure_modes"],
        }
        rows.append(row)
        print(
            f"[run] n={n}: {row['num_successes']}/{row['num_episodes']} = "
            f"{row['success_rate']:.4f} ± {row['success_rate_se']:.4f}  "
            f"CI={[round(x, 3) for x in row['success_rate_ci95']]}  "
            f"{row['failure_modes']}  "
            f"duration={row['duration_s_mean']:.2f}s",
            flush=True,
        )

    # ------------------------------------------------------------------ 汇总表
    print("\n" + "=" * 104)
    print("n_action_steps 控制变量结果（同一 checkpoint / 同一 seeds / 同一评测器）")
    print("=" * 104)
    hdr = (f"{'n_action_steps':>14} {'success':>10} {'rate':>9} {'±SE':>8} "
           f"{'CI95 low':>10} {'CI95 high':>10} {'dur(s)':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['n_action_steps']:>14} "
            f"{str(r['num_successes']) + '/' + str(r['num_episodes']):>10} "
            f"{r['success_rate']:>9.4f} {r['success_rate_se']:>8.4f} "
            f"{r['success_rate_ci95'][0]:>10.4f} {r['success_rate_ci95'][1]:>10.4f} "
            f"{r['duration_s_mean']:>8.2f}"
        )

    base = next((r for r in rows if r["n_action_steps"] == 50), None)
    if base is not None:
        print("\n-- 相对当前基线（n_action_steps=50）--")
        for r in rows:
            if r["n_action_steps"] == 50:
                continue
            delta = r["success_rate"] - base["success_rate"]
            rel = (100.0 * delta / base["success_rate"]) if base["success_rate"] > 0 else float("nan")
            print(
                f"  n={r['n_action_steps']:>3}: rate={r['success_rate']:.4f} vs 50→{base['success_rate']:.4f}  "
                f"绝对差 {delta:+.4f}  相对 {rel:+.0f}%"
            )
    print()

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "ckpt": args.ckpt,
                    "chunk_size": chunk_size,
                    "seeds": args.seeds,
                    "episodes_per_seed": args.episodes,
                    "max_episode_steps": args.max_steps,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[wrote] {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
