"""在 Windows 上运行 ManiSkill ACT baseline（``train_rgbd.py``）的包装脚本。

**为什么需要**（见 ``docs/env_decision.md`` §8）：

1. ManiSkill env 的 ``render_backend`` 默认是 ``"gpu"``（``sapien_cuda``），在本机会让进程
   **原生崩溃**（exit 5、无 traceback），而官方 ``train_rgbd.py`` / ``act/make_env.py``
   构造 env 时不传该字段。
2. ``act/make_env.py`` 只在 ``sim_backend == "physx_cpu"`` 时走 CPU 路径，否则走 GPU 向量化路径。

**做法**：不修改第三方代码，只在启动前 monkeypatch ``gymnasium.make``，把 Windows 必需配置
注入每个 env；其余参数原样透传给 ``train_rgbd.py``。

用法::

    .venv\\Scripts\\python.exe scripts\\train_act_windows.py --env-id PickCube-v1 \\
        --demo-path <...\\trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5> \\
        --sim-backend physx_cpu --no-include-depth --batch-size 32 --total-iters 200 \\
        --num-demos 20 --exp-name act-smoke-rgb
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import gymnasium as gym

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = REPO_ROOT / "third_party" / "ManiSkill-main" / "examples" / "baselines" / "act"

#: 本机（Windows 原生）实测必需的 env 配置
WINDOWS_ENV_DEFAULTS = {
    "render_backend": "cpu",      # 默认 "gpu" 会原生崩溃
    "sim_backend": "physx_cpu",   # Windows 无 GPU 仿真
}

_original_make = gym.make


def _make_with_windows_defaults(env_id, **kwargs):
    for key, value in WINDOWS_ENV_DEFAULTS.items():
        kwargs.setdefault(key, value)
    return _original_make(env_id, **kwargs)


def _patch_act_evaluate() -> None:
    """让 ACT 自带评测失效（改为返回空指标的占位）。

    原因：``act/evaluate.py`` 读取 ``info["final_info"]``，那是 GPU 向量化环境
    （``ManiSkillVectorEnv``）的格式；本机走 CPU 路径（``CPUGymWrapper``）时会 ``KeyError``。
    本项目统一用 ``embodied_core.rollout.runner`` 做评测（保证项目 A/B 结果可直接对照），
    所以这里把 ACT 自带评测替换掉。
    """
    try:
        import act.evaluate as act_evaluate
    except Exception as exc:  # pragma: no cover
        print(f"[warn] 无法 patch act.evaluate：{exc}", file=sys.stderr)
        return
    class _EmptyMetrics(dict):
        """任何 key 都返回空列表，避免 ACT 脚本读取评测字段时 KeyError。"""

        def __missing__(self, key):
            return []

    act_evaluate.evaluate = lambda *args, **kwargs: _EmptyMetrics()


def main() -> int:
    target = ACT_DIR / "train_rgbd.py"
    if not target.exists():
        print(
            f"[error] 未找到 {target}\n"
            "请先获取 ManiSkill 源码到 third_party/ManiSkill-main"
            "（见 docs/env_decision.md 的下载源说明）。",
            file=sys.stderr,
        )
        return 2

    gym.make = _make_with_windows_defaults  # 注入 Windows 必需配置
    sys.path.insert(0, str(ACT_DIR))  # train_rgbd.py 需要 import 同目录的 act 包
    _patch_act_evaluate()  # ACT 自带评测不兼容 CPU 路径 → 用本仓库评测器
    sys.argv = ["train_rgbd.py", *sys.argv[1:]]
    os.chdir(ACT_DIR)  # 官方脚本会把产物写到 ./runs/<exp_name>
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
