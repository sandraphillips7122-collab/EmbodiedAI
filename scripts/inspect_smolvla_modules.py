"""调查 SmolVLA 的真实 module tree，定位可用于 LoRA 的**视觉相关**模块名。

**为什么必须先做这一步**

`lerobot/policies/smolvla/modeling_smolvla.py:420` 的 `_get_default_peft_targets()` 给出的默认 target 是：

```
(model\\.vlm_with_expert\\.lm_expert\\..*\\.(q|v)_proj | model\\.(state_proj|action_in_proj|
 action_out_proj|action_time_mlp_in|action_time_mlp_out))
```

即 **action expert 的 q/v 投影 + 若干投影层** —— **`vlm`（视觉编码器与 LLM 主干）完全不在其中**。
因此**官方默认 LoRA 无法检验"视觉域适配不足"这一假设**，必须用
``--peft.target_modules`` 显式指定视觉模块。

而**具体的模块名不能靠猜**（SmolVLM2 的层级命名与常见 ViT 不同），必须从当前安装版本
的真实 module tree 里读出来 —— 本脚本就是干这个的。

用法::

    .venv\\Scripts\\python.exe scripts/inspect_smolvla_modules.py
    .venv\\Scripts\\python.exe scripts/inspect_smolvla_modules.py --json out.json
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

#: 视觉相关关键词（用于在 module tree 中筛选）
VISION_KEYWORDS = ("vision", "connector", "patch", "image")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打印 SmolVLA 的真实 module tree")
    parser.add_argument("--ckpt", default=str(CKPT))
    parser.add_argument("--json", default=None)
    parser.add_argument("--max-print", type=int, default=120)
    args = parser.parse_args(argv)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"[load] {args.ckpt}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt)

    # 1) 打印 PEFT 的默认 target（直接从代码取，不猜）
    try:
        default_targets = policy._get_default_peft_targets()
    except Exception as exc:  # noqa: BLE001
        default_targets = {"error": repr(exc)}
    print("\n=== `_get_default_peft_targets()` 的官方默认值 ===")
    print(json.dumps(default_targets, ensure_ascii=False, indent=2))

    # 2) 顶层结构
    print("\n=== 顶层子模块 ===")
    for name, mod in policy.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<28} {type(mod).__name__:<32} {n:>14,}")

    # 3) 视觉相关模块（真实名字）
    print("\n=== 名称含 vision/connector/patch/image 的模块 ===")
    vision_mods: list[dict] = []
    for name, mod in policy.named_modules():
        low = name.lower()
        if any(k in low for k in VISION_KEYWORDS):
            n_direct = sum(p.numel() for p in mod.parameters(recurse=False))
            vision_mods.append({"name": name, "type": type(mod).__name__, "params_direct": n_direct})
    for m in vision_mods[: args.max_print]:
        print(f"  {m['name']:<78} {m['type']:<26} {m['params_direct']:>12,}")
    if len(vision_mods) > args.max_print:
        print(f"  ...（共 {len(vision_mods)} 个）")

    # 4) `vlm` 内部的两层结构（便于判断 LoRA 应该挂在哪一层）
    print("\n=== `model.vlm_with_expert.vlm` 的子模块 ===")
    vlm = None
    for name, mod in policy.named_modules():
        if name.endswith("vlm_with_expert.vlm"):
            vlm = (name, mod)
            break
    if vlm:
        base, mod = vlm
        for n2, m2 in mod.named_children():
            n = sum(p.numel() for p in m2.parameters())
            print(f"  {base}.{n2:<34} {type(m2).__name__:<26} {n:>14,}")

    # 5) Linear 层统计（LoRA 只能挂在特定层上）
    print("\n=== 所有 nn.Linear 的名字前缀统计（用于选 target）===")
    import torch.nn as nn

    prefixes: dict[str, int] = {}
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            parts = name.split(".")
            # 取到倒数第二级（如 ...vision_model.encoder.layers.0.self_attn）
            prefix = ".".join(parts[:-1]) if len(parts) > 1 else name
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
    for p, c in sorted(prefixes.items()):
        print(f"  {p:<84} x{c}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"default_targets": default_targets, "vision_modules": vision_mods},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\n[wrote] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
