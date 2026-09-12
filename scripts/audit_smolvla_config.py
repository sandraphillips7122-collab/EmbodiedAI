"""审计 SmolVLA checkpoint 的**真实**训练配置与参数量。

**为什么需要这个脚本**

报告里曾写"450M 总参数 / 99.9M 可训练""无需 LoRA/冻结降级"。
但直接读 checkpoint 的 ``config.json`` 会发现：

    "freeze_vision_encoder": true
    "train_expert_only": true

"我没有主动加降级阶梯" 与 "模型真的做了全参数微调" 是**两回事**。
本脚本用实测把这件事讲清楚：**逐个子模块**打印参数量与 ``requires_grad``，
并明确列出所有可训练参数的归属。

**用法**::

    .venv\\Scripts\\python.exe scripts/audit_smolvla_config.py
    .venv\\Scripts\\python.exe scripts/audit_smolvla_config.py --ckpt <dir> --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: 默认审计两个 checkpoint（v1: 100 eps/5k steps，v2: 500 eps/20k steps）
DEFAULT_CKPTS = {
    "v1_100eps_5k": REPO_ROOT / "experiments/B_smolvla_finetune/checkpoints/005000/pretrained_model",
    "v2_500eps_20k": REPO_ROOT / "experiments/B_smolvla_finetune_v2/checkpoints/020000/pretrained_model",
}

#: config 里与"训练什么"直接相关的字段
CONFIG_KEYS_OF_INTEREST = (
    "freeze_vision_encoder",
    "train_expert_only",
    "train_state_proj",
    "use_peft",
    "chunk_size",
    "n_action_steps",
    "num_steps",
    "max_state_dim",
    "max_action_dim",
    "num_vlm_layers",
    "vlm_model_name",
    "resize_imgs_with_padding",
    "use_cache",
    "attention_mode",
)

#: 需要重点交代归属的模块前缀（其余归入 "other"）
GROUPS = (
    "model.vlm_with_expert",
    "model.vision_model",
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
)


def _fmt(n: int) -> str:
    return f"{n:,}"


def _m(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def audit_one(policy, ckpt: Path) -> dict:
    """统计一个已加载策略的子模块参数量与可训练情况。"""
    cfg = policy.config

    # 1) 顶层子模块（model / …）
    top_rows: list[tuple[str, int, int]] = []
    for name, module in policy.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        top_rows.append((name, total, trainable))

    # 2) 按 GROUPS 指定的前缀归属（自适应：有的模型前缀不同）
    all_params = dict(policy.named_parameters())
    group_stats: dict[str, dict[str, int]] = {g: {"total": 0, "trainable": 0} for g in GROUPS}
    group_stats["other"] = {"total": 0, "trainable": 0}
    for pname, param in all_params.items():
        matched = None
        for g in GROUPS:
            if pname.startswith(g):
                matched = g
                break
        key = matched or "other"
        group_stats[key]["total"] += param.numel()
        if param.requires_grad:
            group_stats[key]["trainable"] += param.numel()

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    # 3) 冻结模块的"名字前缀"去重（按第一层子模块聚合，便于阅读）
    frozen_prefixes: set[str] = set()
    trainable_prefixes: set[str] = set()
    for pname, param in all_params.items():
        parts = pname.split(".")
        prefix = ".".join(parts[:3]) if len(parts) >= 3 else pname
        (trainable_prefixes if param.requires_grad else frozen_prefixes).add(prefix)

    return {
        "ckpt": str(ckpt),
        "config_subset": {k: getattr(cfg, k, "<missing>") for k in CONFIG_KEYS_OF_INTEREST},
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "top_modules": top_rows,
        "group_stats": group_stats,
        "frozen_prefixes": sorted(frozen_prefixes),
        "trainable_prefixes": sorted(trainable_prefixes),
    }


def _print_report(tag: str, r: dict) -> None:
    print("=" * 100)
    print(f"# {tag}")
    print(f"# ckpt: {r['ckpt']}")
    print("=" * 100)

    print("\n-- config 关键字段（直接读自 checkpoint）--")
    for k, v in r["config_subset"].items():
        print(f"  {k:26s} = {v}")

    print("\n-- 总览 --")
    print(f"  总参数      : {_fmt(r['total_params'])}  ({_m(r['total_params'])})")
    print(f"  可训练参数  : {_fmt(r['trainable_params'])}  ({_m(r['trainable_params'])})")
    print(f"  冻结参数    : {_fmt(r['frozen_params'])}  ({_m(r['frozen_params'])})")
    pct = 100.0 * r["trainable_params"] / max(1, r["total_params"])
    print(f"  可训占比    : {pct:.2f}%")

    print("\n-- 顶层子模块 --")
    print(f"  {'module':<28} {'total':>14} {'trainable':>14}  {'status':<10}")
    for name, total, trainable in r["top_modules"]:
        status = "TRAINABLE" if trainable == total and total > 0 else ("FROZEN" if trainable == 0 else "PARTIAL")
        print(f"  {name:<28} {_fmt(total):>14} {_fmt(trainable):>14}  {status:<10}")

    print("\n-- 关键模块归属（实测）--")
    print(f"  {'group':<34} {'total':>14} {'trainable':>14}  {'status':<10}")
    for g, st in r["group_stats"].items():
        if st["total"] == 0:
            continue
        status = "TRAINABLE" if st["trainable"] == st["total"] else ("FROZEN" if st["trainable"] == 0 else "PARTIAL")
        print(f"  {g:<34} {_fmt(st['total']):>14} {_fmt(st['trainable']):>14}  {status:<10}")

    print("\n-- 可训练参数名前缀（去重后，前 20 个）--")
    for p in r["trainable_prefixes"][:20]:
        print(f"  + {p}")
    if len(r["trainable_prefixes"]) > 20:
        print(f"  ...（共 {len(r['trainable_prefixes'])} 个前缀）")

    print("\n-- 冻结参数名前缀（去重后，前 20 个）--")
    for p in r["frozen_prefixes"][:20]:
        print(f"  - {p}")
    if len(r["frozen_prefixes"]) > 20:
        print(f"  ...（共 {len(r['frozen_prefixes'])} 个前缀）")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计 SmolVLA checkpoint 的真实训练配置与参数量")
    parser.add_argument("--ckpt", action="append", default=[], help="checkpoint 目录（可多次指定）")
    parser.add_argument("--json", default=None, help="把审计结果写成 json")
    args = parser.parse_args(argv)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    targets: dict[str, Path] = {}
    if args.ckpt:
        for i, c in enumerate(args.ckpt):
            targets[f"ckpt{i}"] = Path(c)
    else:
        targets = dict(DEFAULT_CKPTS)

    results: dict[str, dict] = {}
    for tag, ckpt in targets.items():
        if not ckpt.is_dir():
            print(f"[skip] {tag}: 目录不存在 -> {ckpt}")
            continue
        print(f"[load] {tag} ...", flush=True)
        policy = SmolVLAPolicy.from_pretrained(str(ckpt))
        r = audit_one(policy, ckpt)
        results[tag] = r
        _print_report(tag, r)

        # 及时释放，避免同时驻留两个 450M 模型
        del policy
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    if args.json:
        Path(args.json).write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"[wrote] {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
