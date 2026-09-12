"""2×2 析因评测：**语言条件性**的判定实验。

## 设计

把"场景"与"指令"当成两个独立因子做交叉：

| 格 | 场景 ``target_color`` | 传给策略的指令 | 含义 |
|---|---|---|---|
| 1 | red  | "pick up the **red** cube ..."  | ✅ 正确指令 |
| 2 | red  | "pick up the **blue** cube ..." | ❌ 错误指令 |
| 3 | blue | "pick up the **blue** cube ..." | ✅ 正确指令 |
| 4 | blue | "pick up the **red** cube ..."  | ❌ 错误指令 |

**场景固定的是物理布局**（哪一块在槽位 A）；**指令**是给模型的文字。
任务定义是"把**指令指定的那一块**送到 goal"，所以：

- **正确指令**格：模型只要按指示抓 A 位置的块 → 与训练时一致
- **错误指令**格：模型被要求抓 B 位置那一块，而**示教里从没教过抓 B** → 必然失败（若它真读指令）

## 判据与零假设

**零假设 H0**：模型忽略语言 → 同一场景下"正确指令"与"错误指令"的成功率**无差异**。

- 若 `rate(正确) − rate(错误)` 显著 > 0 → 语言条件性成立（模型确实读了指令）
- 若无显著差异 → **模型忽略语言**（这是同样有价值的结论，如实报告，不做美化）

另外记录 ``is_wrong_obj_placed``（错误方块被送到 goal）作为辅助信号：
若模型忽略指令，它会按训练习惯去抓 A 块 —— 在错误指令下这就变成"送错了块"。

用法::

    .venv\\Scripts\\python.exe scripts/langcond_eval.py --episodes 10 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT = REPO_ROOT / "experiments/B_langcond_finetune/checkpoints/006000/pretrained_model"
INSTR_RED = "pick up the red cube and move it to the goal position"
INSTR_BLUE = "pick up the blue cube and move it to the goal position"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="语言条件性 2×2 析因评测")
    parser.add_argument("--ckpt", default=str(CKPT))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=10, help="每个 seed 块的集数")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments/B_langcond_summary.json"))
    args = parser.parse_args(argv)

    from embodied_core.envs.adapter import EnvSpec
    from embodied_core.rollout.runner import RolloutConfig, run_rollout
    from embodied_core.rollout.smolvla_policy import SmolVLAPolicyAdapter
    import embodied_core.envs.dual_cube  # noqa: F401  注册 PickDualCube-v1

    print(f"[setup] 加载 SmolVLA（只加载一次，逐格只切换 instruction）: {args.ckpt}", flush=True)
    adapter = SmolVLAPolicyAdapter(args.ckpt, instruction=INSTR_RED, device="cuda")

    cells = [
        ("red", "red", INSTR_RED, True),
        ("red", "blue", INSTR_BLUE, False),
        ("blue", "blue", INSTR_BLUE, True),
        ("blue", "red", INSTR_RED, False),
    ]

    rows = []
    for scene, instr_tag, instruction, is_correct in cells:
        # 复用同一策略实例，只切 instruction（重建实例会在第二档触发 TDR）
        adapter.instruction = instruction if instruction.endswith("\n") else instruction + "\n"
        adapter.policy.reset()

        exp_id = f"B_langcond_scene{scene}_instr{instr_tag}"
        print(
            f"\n[run] scene={scene}  instruction={instr_tag}  "
            f"({'正确' if is_correct else '错误'})  "
            f"seeds={args.seeds} x {args.episodes}",
            flush=True,
        )
        cfg = RolloutConfig(
            exp_id=exp_id,
            env=EnvSpec(
                task="PickDualCube-v1",
                max_episode_steps=args.max_steps,
                extra={"target_color": scene},
            ),
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
            "scene": scene,
            "instruction": instr_tag,
            "is_correct_instruction": is_correct,
            "exp_id": exp_id,
            "num_episodes": summary["num_episodes"],
            "num_successes": summary["num_successes"],
            "success_rate": summary["success_rate"],
            "success_rate_se": summary["success_rate_se"],
            "success_rate_ci95": summary["success_rate_ci95"],
            "failure_modes": summary["failure_modes"],
        }
        rows.append(row)
        print(
            f"[run] scene={scene} instr={instr_tag}: "
            f"{row['num_successes']}/{row['num_episodes']} = {row['success_rate']:.4f} "
            f"± {row['success_rate_se']:.4f}  CI={[round(x, 3) for x in row['success_rate_ci95']]}  "
            f"{row['failure_modes']}",
            flush=True,
        )

    # ------------------------------------------------------------------ 汇总
    print("\n" + "=" * 96)
    print("语言条件性 2x2 析因结果")
    print("=" * 96)
    hdr = f"{'scene':>6} {'instr':>6} {'correct?':>9} {'success':>10} {'rate':>9} {'±SE':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['scene']:>6} {r['instruction']:>6} "
            f"{('YES' if r['is_correct_instruction'] else 'NO'):>9} "
            f"{str(r['num_successes']) + '/' + str(r['num_episodes']):>10} "
            f"{r['success_rate']:>9.4f} {r['success_rate_se']:>8.4f}"
        )

    print("\n-- 关键对比：同场景下 正确指令 vs 错误指令 --")
    for scene in ("red", "blue"):
        corr = next(r for r in rows if r["scene"] == scene and r["is_correct_instruction"])
        wrong = next(r for r in rows if r["scene"] == scene and not r["is_correct_instruction"])
        diff = corr["success_rate"] - wrong["success_rate"]
        # 差值标准误（两独立比例）
        se = (corr["success_rate_se"] ** 2 + wrong["success_rate_se"] ** 2) ** 0.5
        print(
            f"  scene={scene}: 正确 {corr['success_rate']:.4f}  vs  错误 {wrong['success_rate']:.4f}  "
            f"→ 差值 {diff:+.4f} ± {se:.4f}"
        )

    overall_corr = [r["success_rate"] for r in rows if r["is_correct_instruction"]]
    overall_wrong = [r["success_rate"] for r in rows if not r["is_correct_instruction"]]
    mc = sum(overall_corr) / len(overall_corr)
    mw = sum(overall_wrong) / len(overall_wrong)
    print(f"\n  汇总：正确指令均值 {mc:.4f}  vs  错误指令均值 {mw:.4f}  → 差 {mc - mw:+.4f}")
    if mc - mw <= 0:
        print("  判读：**未观察到语言条件性**（正确指令并不比错误指令好）")
        print("        → 按预设判据，应如实报告为「模型忽略语言」，而不是包装成部分理解。")
    else:
        print("  判读：正确指令优于错误指令，提示存在语言条件性（需结合样本量看 CI）。")
    print()

    Path(args.out).write_text(
        json.dumps({"ckpt": args.ckpt, "seeds": args.seeds,
                    "episodes_per_seed": args.episodes, "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
