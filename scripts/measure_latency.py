"""拆分测量 SmolVLA 的推理延迟与闭环延迟。

**为什么必须重测**

报告里曾写"策略延迟 mean 6.7 ms ≈ 149 Hz"。这个数字是**闭环 step latency 的平均值** ——
而 SmolVLA 的 ``select_action`` 自带 **action queue**（``n_action_steps=50``）：
真正做 VLA forward 只在 **每 50 步中的 1 步**，其余 49 步只是从队列里取出预先算好的动作。

因此：
- "6.7 ms / step" 是 **平均闭环延迟**，不是推理速度；
- 把队列平均后的数字当作 "VLA FPS" 会**严重高估**模型的实时能力；
- 真正决定闭环是否跟得上的是 **forward 延迟**（forward 发生时，那一步会明显变慢）。

本脚本把延迟拆成六项并分别报分位数：

1. ``preprocessor``  —— 特征重命名 / 归一化 / 语言 tokenize
2. **真正 forward**  —— queue 为空时的 ``select_action``（含 10 步 flow-matching 去噪）
3. **queue 出队**    —— queue 非空时的 ``select_action``
4. ``postprocessor`` —— 动作反归一化
5. ``env.step``      —— 仿真 + 渲染
6. **完整闭环 step** —— obs → pre → select_action → post → env.step

用法::

    .venv\\Scripts\\python.exe scripts/measure_latency.py
    .venv\\Scripts\\python.exe scripts/measure_latency.py --ckpt <dir> --steps 120 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CKPT = (
    REPO_ROOT / "experiments/B_smolvla_finetune_v2/checkpoints/020000/pretrained_model"
)
INSTRUCTION = "pick up the red cube and move it to the goal position"


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()


def _pct(xs: list[float], q: float) -> float:
    """简单分位数（线性插值），避免依赖 numpy 版本差异。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {}
    return {
        "n": len(xs),
        "mean_ms": 1000 * statistics.fmean(xs),
        "p50_ms": 1000 * _pct(xs, 0.50),
        "p95_ms": 1000 * _pct(xs, 0.95),
        "min_ms": 1000 * min(xs),
        "max_ms": 1000 * max(xs),
    }


def _fmt(tag: str, st: dict) -> str:
    if not st:
        return f"  {tag:<34} (no samples)"
    return (
        f"  {tag:<34} n={st['n']:>4}  mean={st['mean_ms']:>8.2f}ms  "
        f"p50={st['p50_ms']:>8.2f}  p95={st['p95_ms']:>8.2f}  "
        f"min={st['min_ms']:>7.2f}  max={st['max_ms']:>8.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拆分测量 SmolVLA 延迟")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--steps", type=int, default=120, help="闭环测的步数（建议 >= 2*n_action_steps）")
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    from embodied_core.envs.adapter import EnvSpec, make_env
    from embodied_core.rollout.smolvla_policy import SmolVLAPolicyAdapter

    device = torch.device("cuda")

    print("[setup] 构建 3 相机环境 ...", flush=True)
    env = make_env(EnvSpec(max_episode_steps=args.steps + 10), cameras="three")
    obs, _ = env.reset(seed=0)

    print(f"[setup] 加载 SmolVLA: {args.ckpt}", flush=True)
    adapter = SmolVLAPolicyAdapter(args.ckpt, instruction=args.instruction, device="cuda")
    policy = adapter.policy
    pre, post = adapter.preprocessor, adapter.postprocessor

    def _queue_len() -> int:
        """探测 lerobot 内部的 action queue 长度（字段名可能随版本变化）。"""
        for attr in ("_queues", "queues"):
            q = getattr(policy, attr, None)
            if q is not None:
                try:
                    return len(q)
                except TypeError:
                    pass
        return -1

    results: dict[str, dict] = {}

    # ---------------------------------------------------------------- 1) 预/后处理
    pre_times, post_times = [], []
    batch = pre(adapter._build_obs(obs, None))
    _ = pre(adapter._build_obs(obs, None))
    for _ in range(30):
        _sync(device)
        t0 = time.perf_counter()
        b = pre(adapter._build_obs(obs, None))
        _sync(device)
        pre_times.append(time.perf_counter() - t0)

        _sync(device)
        t0 = time.perf_counter()
        _ = post(torch.zeros(1, 8, device=device))
        _sync(device)
        post_times.append(time.perf_counter() - t0)
    results["preprocessor"] = _stats(pre_times)
    results["postprocessor"] = _stats(post_times)

    # -------------------------------------------------- 2/3) forward vs queue 出队
    # 关键：每次 reset() 会清空 action queue，随后的第一次 select_action 必然真正 forward，
    # 其后的 n_action_steps-1 次只是出队。
    n_action_steps = int(getattr(policy.config, "n_action_steps", 50))
    print(f"[probe] n_action_steps = {n_action_steps}", flush=True)

    forward_times, dequeue_times = [], []
    queue_lens: list[int] = []
    # 重复 12 轮：每轮 reset() 后第一次必然真正 forward，其余 49 次只是出队。
    # forward 样本数 = 12（初版只跑 4 轮，样本偏少且 forward 波动很大）。
    for _rep in range(12):
        policy.reset()
        for _i in range(n_action_steps):
            _sync(device)
            t0 = time.perf_counter()
            _ = policy.select_action(batch)
            _sync(device)
            dt = time.perf_counter() - t0
            qlen = _queue_len()
            queue_lens.append(qlen)
            if _i == 0:
                forward_times.append(dt)
            else:
                dequeue_times.append(dt)
    results["forward_queue_empty"] = _stats(forward_times)
    results["dequeue_queue_nonempty"] = _stats(dequeue_times)
    results["_queue_len_samples"] = {"values": sorted(set(queue_lens))[:10]}

    # ---------------------------------------------------------------------- 5) env
    env_times = []
    for _ in range(min(args.steps, 60)):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        _sync(device)
        t0 = time.perf_counter()
        obs, _r, term, trunc, _info = env.step(a)
        _sync(device)
        env_times.append(time.perf_counter() - t0)
        if term or trunc:
            obs, _ = env.reset(seed=0)
    results["env_step"] = _stats(env_times)

    # ------------------------------------------------------------ 6) 完整闭环 step
    obs, _ = env.reset(seed=1)
    adapter.reset(seed=1)
    loop_times = []
    step_lat = []
    for i in range(args.steps):
        _sync(device)
        t0 = time.perf_counter()
        action = adapter.predict(obs)
        step_lat.append(time.perf_counter() - t0)

        _sync(device)
        t0 = time.perf_counter()
        obs, _r, term, trunc, _info = env.step(action)
        _sync(device)
        loop_times.append(step_lat[-1] + (time.perf_counter() - t0))
        if term or trunc:
            obs, _ = env.reset(seed=1)
            adapter.reset(seed=1)
    results["closed_loop_step"] = _stats(loop_times)
    results["policy_call_within_loop"] = _stats(step_lat)

    # ------------------------------------------------------------------------ 报告
    print("\n" + "=" * 112)
    print("SmolVLA 延迟拆分（Windows 原生 / RTX 4060 Laptop / render_backend=cpu）")
    print("=" * 112)
    print(_fmt("1. preprocessor", results["preprocessor"]))
    print(_fmt("2. FORWARD (queue empty)", results["forward_queue_empty"]))
    print(_fmt("3. dequeue (queue non-empty)", results["dequeue_queue_nonempty"]))
    print(_fmt("4. postprocessor", results["postprocessor"]))
    print(_fmt("5. env.step (sim + render)", results["env_step"]))
    print(_fmt("6. closed-loop step", results["closed_loop_step"]))
    print(_fmt("   (6a) policy call in loop", results["policy_call_within_loop"]))

    fwd = results["forward_queue_empty"].get("mean_ms", float("nan"))
    deq = results["dequeue_queue_nonempty"].get("mean_ms", float("nan"))
    loop = results["closed_loop_step"].get("mean_ms", float("nan"))
    envms = results["env_step"].get("mean_ms", float("nan"))

    print("\n-- 派生结论 --")
    if fwd == fwd and deq == deq and n_action_steps > 0:
        avg_policy = (fwd + deq * (n_action_steps - 1)) / n_action_steps
        print(f"  单步平均 policy 耗时（按队列权重折算） = {avg_policy:.2f} ms")
        if avg_policy > 0:
            print(f"  折算等效 policy 频率                    = {1000.0 / avg_policy:.1f} Hz")
        print(f"  真正 forward 延迟                        = {fwd:.2f} ms "
              f"（单次，含 10 步 flow-matching 去噪）")
        print(f"  纯出队延迟                               = {deq:.2f} ms")
        print(f"  forward 触发频率                         = 每 {n_action_steps} 步 1 次")
    print(f"  闭环平均 step latency                    = {loop:.2f} ms  ({1000.0 / loop:.1f} Hz)"
          if loop == loop and loop > 0 else "  闭环平均 step latency                    = n/a")
    print(f"  其中环境 step（仿真+渲染，CPU 渲染）      = {envms:.2f} ms")
    if loop == loop and envms == envms and loop > 0:
        print(f"  环境占闭环比例                           = {100.0 * envms / loop:.1f}%")

    print("\n-- 队列探测 --")
    print(f"  n_action_steps (config) = {n_action_steps}")
    print(f"  observed queue length samples = {results['_queue_len_samples']['values']}")
    print()

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "ckpt": args.ckpt,
                    "steps": args.steps,
                    "n_action_steps": n_action_steps,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[wrote] {args.json}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
