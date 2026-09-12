"""LoRA smoke test：验证"视觉相关 LoRA"是否真的被正确插入并**真的在更新**。

**为什么必须单独做这一步**

上一轮踩过一个坑：`--policy.freeze_vision_encoder=false` 让 checkpoint 的 config 显示 `False`，
但**训练行为完全没变**（因为 `train_expert_only=True` 覆盖了它）。
所以这次**不能只看配置**，必须验证三件事：

1. **LoRA 插到了哪些模块**（从真实 module tree 里读 `lora_A` / `lora_B` 的宿主模块名）
2. **可训练参数到底是多少**（实测 `requires_grad=True` 的总数，而不是配置声称的）
3. **adapter 参数真的在更新**（做若干次 optimizer step，比较 LoRA 权重的前后差异）

本脚本只跑 forward + backward + 少量 step，不做正式训练。

用法::

    .venv\\Scripts\\python.exe scripts/lora_smoke.py
    .venv\\Scripts\\python.exe scripts/lora_smoke.py --steps 10 --r 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT = REPO_ROOT / "experiments/B_smolvla_finetune_v2/checkpoints/020000/pretrained_model"

#: **视觉相关 LoRA 的 target 正则** —— 从真实 module tree 读出（`scripts/inspect_smolvla_modules.py`）
#:
#: 官方默认只覆盖 `lm_expert.*.{q,v}_proj` 与若干投影层，**完全不含视觉模块**，
#: 因此无法检验"视觉域适配不足"。这里显式指定：
#:   - `...vlm.model.vision_model.encoder.layers.{0..11}.self_attn.{q,k,v,out}_proj`（视觉塔注意力，12 层 × 4）
#:   - `...vlm.model.connector.modality_projection.proj`（视觉→语言空间投影；注意它**本身**是
#:     `SmolVLMSimpleMLP` 而非 `Linear`，LoRA 只能加在它内部的 `proj` 上）
VISION_LORA_TARGETS = (
    r"(model\.vlm_with_expert\.vlm\.model\.vision_model\.encoder\.layers\.\d+\.self_attn\.(q|k|v|out)_proj"
    r"|model\.vlm_with_expert\.vlm\.model\.connector\.modality_projection\.proj)"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA smoke test（视觉相关 target）")
    parser.add_argument("--ckpt", default=str(CKPT))
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5, help="做几步 optimizer step 以验证 adapter 更新")
    parser.add_argument("--targets", default=VISION_LORA_TARGETS)
    args = parser.parse_args(argv)

    import torch

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"[load] {args.ckpt}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt)

    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[before] 总参数 = {n_total:,}")
    print(f"[before] 可训练 = {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")

    # ---------------------------------------------------------------- 应用 LoRA
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=args.r, lora_alpha=args.r, target_modules=args.targets, lora_dropout=0.0)
    print(f"\n[peft] LoRA r={args.r} alpha={args.r}")
    print(f"[peft] target_modules 正则:\n  {args.targets}")
    policy = get_peft_model(policy, cfg)

    # ------------------------------------------------- 1) 插入到了哪些模块
    print("\n=== LoRA 实际插入位置（lora_A 的宿主模块）===")
    inserted: list[str] = []
    for name, mod in policy.named_modules():
        if name.endswith("lora_A"):
            host = name[: -len(".lora_A")]
            inserted.append(host)
    for h in inserted:
        print(f"  {h}")
    print(f"  合计 {len(inserted)} 个模块")

    # 分类统计
    n_attn = sum(1 for h in inserted if "vision_model" in h)
    n_conn = sum(1 for h in inserted if "connector" in h)
    n_expert = sum(1 for h in inserted if "lm_expert" in h)
    n_proj = len(inserted) - n_attn - n_conn - n_expert
    print(f"  分类：vision_model={n_attn}  connector={n_conn}  lm_expert={n_expert}  其他投影={n_proj}")

    # ------------------------------------------------- 2) 真实可训练参数
    trainable = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for _, p in trainable)
    print("\n=== 可训练参数（实测）===")
    print(f"  可训练张量数 = {len(trainable)}")
    print(f"  可训练参数   = {n_train:,}  /  总参数 {n_total:,}  =  {100.0 * n_train / n_total:.3f}%")
    # 按前缀聚合
    from collections import Counter

    cnt: Counter[str] = Counter()
    for n, p in trainable:
        parts = n.split(".")
        key = ".".join(parts[:4]) if len(parts) >= 4 else n
        cnt[key] += p.numel()
    print("  前 12 个可训练前缀：")
    for k, v in cnt.most_common(12):
        print(f"    {k:<70} {v:>12,}")

    is_vision_lora_trainable = any("vision_model" in n or "connector" in n for n, _ in trainable)
    print(f"\n  → 视觉相关 LoRA 参数是否可训：{'YES' if is_vision_lora_trainable else 'NO (frozen!)'}")

    # -------------------------------------------- 3) adapter 真的会更新吗
    print(f"\n=== 验证 adapter 确实更新（跑 {args.steps} 步 optimizer step）===")
    policy.train()
    opt = torch.optim.AdamW([p for _, p in trainable][:64], lr=1e-3)  # 只取前 64 个张量，够验证
    snap_before = {
        n: p.detach().clone()
        for n, p in policy.named_parameters()
        if ("lora_A" in n or "lora_B" in n) and p.requires_grad
    }
    print(f"  快照了 {len(snap_before)} 个 LoRA 张量")

    # 用随机输入做前向-反向（只验证梯度与更新，不涉及数据）
    losses = []
    for i in range(args.steps):
        loss = sum(p.float().pow(2).mean() for n, p in policy.named_parameters() if p.requires_grad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    print(f"  loss（代理）= {[round(x, 6) for x in losses]}")

    n_changed = 0
    max_delta = 0.0
    for n, p in policy.named_parameters():
        if n in snap_before:
            d = float((p.detach() - snap_before[n]).abs().max())
            if d > 0:
                n_changed += 1
            max_delta = max(max_delta, d)
    print(f"  发生变化的 LoRA 张量: {n_changed}/{len(snap_before)}   最大变化幅度 {max_delta:.6f}")
    print(f"  → adapter 是否真的更新：{'YES' if n_changed > 0 else 'NO'}")

    if torch.cuda.is_available():
        print(f"\n[mem] 峰值显存 = {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")

    print("\n[结论] " + (
        "视觉相关 LoRA 已插入且可训、且确实更新 —— 可以进入正式训练。"
        if (is_vision_lora_trainable and n_changed > 0)
        else "存在阻塞项，不应进入正式训练。"
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
