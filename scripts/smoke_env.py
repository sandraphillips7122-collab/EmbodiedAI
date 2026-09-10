"""S0 冒烟测试 · ManiSkill 环境构建与参数诊断

背景：在 Windows 原生环境下，`python -m mani_skill.trajectory.replay_trajectory`
以 `obs_mode=rgb` 构建 PickCube-v1 时报
`TypeError: 'NoneType' object is not callable ... for PickCube-v1`。
本脚本用最小组合逐个排除参数，并把**完整 traceback** 打印出来定位根因。

用法::

    .venv\\Scripts\\python.exe scripts\\smoke_env.py

退出码 0 表示所有组合都成功。
"""

from __future__ import annotations

import traceback

import gymnasium as gym

import mani_skill  # noqa: F401  # 导入以触发环境注册

TASK = "PickCube-v1"
BASE_KWARGS = dict(sim_backend="physx_cpu", num_envs=1)


def attempt(label: str, **kwargs) -> bool:
    try:
        env = gym.make(TASK, **BASE_KWARGS, **kwargs)
        print(f"[OK]   {label}")
        env.close()
        return True
    except Exception:
        print(f"[FAIL] {label}")
        print(traceback.format_exc())
        return False


def main() -> int:
    results: dict[str, bool] = {}
    results["state 基线"] = attempt("state 基线", obs_mode="state")
    results["rgb"] = attempt("rgb", obs_mode="rgb")
    results["state + render_mode"] = attempt(
        "state + render_mode", obs_mode="state", render_mode="rgb_array"
    )
    results["rgb + control_mode"] = attempt(
        "rgb + control_mode", obs_mode="rgb", control_mode="pd_ee_delta_pos"
    )
    results["state + shader_dir"] = attempt(
        "state + shader_dir", obs_mode="state", shader_dir="default"
    )

    print("\n=== 汇总 ===")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
