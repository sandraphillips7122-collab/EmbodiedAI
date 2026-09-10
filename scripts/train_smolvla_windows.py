"""在 Windows 上跑 SmolVLA 微调（lerobot-train）的包装脚本。

解决三个 Windows / interop 特有问题（都已在实测中踩到）：

1. **收尾建符号链接失败**：`lerobot-train` 结束后会给最新 checkpoint 建 `checkpoints/last`
   符号链接，而 Windows 默认无该特权（`OSError: [WinError 1314] 客户端没有所需的特权`），
   导致**训练明明跑完却以异常结束**。这里把 `update_last_checkpoint` 换成 no-op
   （checkpoint 目录本身已经保存好了）。
2. **`--policy.path` 被路径转换**：interop 会把 `--policy.path=lerobot/smolvla_base` 里的 `/`
   变成 `\`，导致 HF repo id 校验失败（`HFValidationError: ... 'lerobot\\smolvla_base'`）。
   → 请直接传**本地快照的绝对路径**（Windows 风格，例如
   `C:/Users/<user>/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/<hash>`）。
3. **必须显式关闭 push_to_hub**：否则 `TrainConfig.validate()` 会因为缺少 `repo_id` 报错
   （`--policy.push_to_hub=false`）。

用法::

    .venv\\Scripts\\python.exe scripts\\train_smolvla_windows.py ^
        --policy.path="C:/Users/29446/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/<hash>" ^
        --policy.push_to_hub=false ^
        --dataset.repo_id=local/pickcube_3cam_100 ^
        --dataset.root="C:/Users/29446/.cache/huggingface/lerobot/local/pickcube_3cam_100" ^
        --batch_size=8 --steps=10000 --save_freq=2000 --log_freq=50 ^
        --num_workers=0 --wandb.enable=false ^
        --output_dir="E:/EmbodiedAI/experiments/B_smolvla_finetune"
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _disable_last_checkpoint_symlink() -> None:
    """把 leaky symlink 步骤替换为 no-op（Windows 无 symlink 特权）。"""
    import lerobot.common.train_utils as train_utils

    def _noop(checkpoint_dir):
        print(f"[train_smolvla_windows] 跳过 symlink；checkpoint 已保存在 {checkpoint_dir}")

    train_utils.update_last_checkpoint = _noop

    # lerobot_train.py 是通过 from ... import update_last_checkpoint 引入的，需同时改其模块属性
    import lerobot.scripts.lerobot_train as lt

    if hasattr(lt, "update_last_checkpoint"):
        lt.update_last_checkpoint = _noop


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    _disable_last_checkpoint_symlink()

    # 注意：通过 runpy 执行官方入口，保证 argv 原样透传
    sys.argv = ["lerobot-train", *sys.argv[1:]]
    import os

    os.chdir(repo_root)
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
