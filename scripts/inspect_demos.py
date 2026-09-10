"""统计 ManiSkill 示教数据集，产出「数据卡」。

用途：把「数据→训练→评测」里的**数据侧**也变成可复核的证据（分布、覆盖、动作范围），
而不是只有一句「用了 100 条 demo」。纯 h5py/numpy，不依赖 lerobot。

用法::

    .venv\\Scripts\\python.exe scripts\\inspect_demos.py \\
        --h5 "C:\\Users\\29446\\.maniskill\\demos\\PickCube-v1\\motionplanning\\trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5" \\
        --out docs/data_card_pickcube_rgb.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import h5py
import numpy as np


def _describe_struct(value, depth: int = 0, max_depth: int = 3) -> str:
    """把 h5 里的嵌套结构描述成短字符串。"""
    if isinstance(value, h5py.Dataset):
        return f"Dataset{tuple(value.shape)} {value.dtype}"
    if isinstance(value, h5py.Group):
        if depth >= max_depth:
            return f"Group(keys={list(value.keys())})"
        inner = ", ".join(
            f"{k}: {_describe_struct(v, depth + 1, max_depth)}" for k, v in value.items()
        )
        return f"{{{inner}}}"
    return repr(value)


def summarize(path: Path, max_trajectories: int) -> dict:
    with h5py.File(path, "r") as f:
        traj_names = sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))
        selected = traj_names[:max_trajectories]

        first = f[selected[0]]
        structure = {k: _describe_struct(v) for k, v in first.items()}
        attrs = {k: str(v) for k, v in first.attrs.items()}

        lengths: list[int] = []
        successes: list[bool] = []
        action_chunks: list[np.ndarray] = []

        for name in selected:
            grp = f[name]
            actions = np.asarray(grp["actions"])
            action_chunks.append(actions)
            lengths.append(int(actions.shape[0]))
            if "success" in grp:
                succ = np.asarray(grp["success"])
                successes.append(bool(succ[-1]) if succ.size else False)

        actions_all = np.concatenate(action_chunks, axis=0)

    return {
        "num_trajectories_total": len(traj_names),
        "num_trajectories_inspected": len(selected),
        "episode_length": {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(statistics.fmean(lengths)),
            "median": float(statistics.median(lengths)),
        },
        "success_rate_inspected": float(np.mean(successes)) if successes else None,
        "action_dim": int(actions_all.shape[-1]),
        "action_stats": {
            "min": np.round(actions_all.min(axis=0), 4).tolist(),
            "max": np.round(actions_all.max(axis=0), 4).tolist(),
            "mean": np.round(actions_all.mean(axis=0), 4).tolist(),
            "std": np.round(actions_all.std(axis=0), 4).tolist(),
        },
        "h5_structure": structure,
        "episode_attrs": attrs,
    }


def render_markdown(info: dict, source: str) -> str:
    ep = info["episode_length"]
    lines = [
        "# 数据卡（ManiSkill 示教）",
        "",
        f"- 数据文件：`{source}`",
        f"- 轨迹总数：**{info['num_trajectories_total']}**（本卡片统计前 {info['num_trajectories_inspected']} 条）",
        f"- episode 长度：min={ep['min']} / median={ep['median']:.0f} / mean={ep['mean']:.1f} / max={ep['max']}",
        f"- 抽样成功率：{info['success_rate_inspected']}",
        f"- 动作维度：**{info['action_dim']}**",
        "",
        "## 动作统计（全量抽样）",
        "",
        "| 维度 | min | max | mean | std |",
        "|---|---|---|---|---|",
    ]
    stats = info["action_stats"]
    for i in range(info["action_dim"]):
        lines.append(
            f"| {i} | {stats['min'][i]} | {stats['max'][i]} | {stats['mean'][i]} | {stats['std'][i]} |"
        )
    lines += ["", "## h5 结构（第一条轨迹）", "", "```json"]
    lines.append(json.dumps(info["h5_structure"], ensure_ascii=False, indent=2))
    lines += ["```", ""]
    if info["episode_attrs"]:
        lines += ["## 轨迹属性", "", "```json", json.dumps(info["episode_attrs"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ManiSkill 示教数据集统计（数据卡生成）")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", default=None, help="Markdown 输出路径（默认仅打印到 stdout）")
    parser.add_argument("--max-trajectories", type=int, default=100)
    args = parser.parse_args()

    path = Path(args.h5)
    info = summarize(path, args.max_trajectories)
    print(json.dumps({k: v for k, v in info.items() if k != "h5_structure"}, ensure_ascii=False, indent=2))
    print("h5 structure:", json.dumps(info["h5_structure"], ensure_ascii=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(info, str(path)), encoding="utf-8")
        print(f"[inspect] 数据卡已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
