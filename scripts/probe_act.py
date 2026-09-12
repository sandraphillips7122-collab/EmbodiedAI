"""ACT 策略动作探针：判断推理是否给出合理动作。

用途：当评测成功率为 0 时，区分「策略真的没学会」与「推理链路有 bug」——
观察动作是否非零、是否随步数变化、reward / success 是否在推进。

用法::

    .venv\\Scripts\\python.exe scripts\\probe_act.py --ckpt <.../10000.pt> --steps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许从任意工作目录运行：Python 只把「脚本所在目录」加入 sys.path，
# 因此在仓库根执行 `python scripts/probe_act.py` 时会找不到 embodied_core 包。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embodied_core.envs.adapter import EnvSpec, make_env
from embodied_core.rollout.act_policy import ActPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description="ACT 策略动作探针")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    env = make_env(EnvSpec())
    policy = ActPolicy(args.ckpt, device=args.device, max_timesteps=200)
    obs, _ = env.reset(seed=args.seed)
    policy.reset(seed=args.seed)

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    total_reward = 0.0
    for step in range(args.steps):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        print(f"step {step:02d} action={action} success={info.get('success')} reward={float(reward):.3f}")
        if terminated or truncated:
            print(f"episode ended at step {step}")
            break

    print("total_reward:", round(total_reward, 3))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
