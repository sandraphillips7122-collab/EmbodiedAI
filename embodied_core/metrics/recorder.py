"""实验指标记录：episode 级 JSONL + 成功率统计。

设计原则（对应 README 的诚实性约定）：
- 每个对外汇报的数字都必须能追溯到 ``experiments/<exp_id>/episodes.jsonl`` 里的一条记录。
- 本模块**不依赖** ManiSkill / torch / numpy，纯标准库，便于离线分析与自查。

统计口径：
- ``rate``      = successes / n（二项比例）
- ``se``        = sqrt(p(1-p)/n)，比例的标准误
- ``ci95``      = Wilson score interval（小样本比正态近似更稳健，50–100 次 rollout 时差异明显）

用法::

    python -m embodied_core.metrics.recorder --selftest    # 内置自查
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# 失败模式标签：由评测器在 rollout 结束时判定，禁止事后手工编造
FAILURE_MODES = (
    "success",
    "no_contact",   # 未接触到目标物体
    "slip",         # 抓取后滑落
    "collision",    # 碰撞/推飞目标
    "timeout",      # 超时未完成
    "not_reached",  # 未到达目标位姿
    "unknown",      # 尚未分类
)


@dataclass(frozen=True)
class EpisodeResult:
    """一次 rollout 的结果（一行 JSONL）。"""

    exp_id: str
    task: str
    policy: str
    seed: int
    episode_idx: int
    success: bool
    steps: int
    duration_s: float
    failure_mode: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.failure_mode not in FAILURE_MODES:
            raise ValueError(
                f"unknown failure_mode={self.failure_mode!r}; expected one of {FAILURE_MODES}"
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class SuccessRate:
    """成功率及其不确定度。"""

    n: int
    successes: int
    rate: float
    se: float
    ci95_low: float
    ci95_high: float

    def __str__(self) -> str:
        return (
            f"{self.successes}/{self.n} = {self.rate:.3f} "
            f"(±{self.se:.3f} SE, 95% CI [{self.ci95_low:.3f}, {self.ci95_high:.3f}])"
        )


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval（95% 默认）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denominator
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(successes: Sequence[bool]) -> SuccessRate:
    """把一串布尔结果汇总为成功率 ± 不确定度。空输入返回 nan（显式暴露，不默认成 0）。"""
    n = len(successes)
    k = sum(1 for s in successes if s)
    if n == 0:
        return SuccessRate(0, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    rate = k / n
    se = math.sqrt(rate * (1 - rate) / n)
    low, high = _wilson_interval(k, n)
    return SuccessRate(n=n, successes=k, rate=rate, se=se, ci95_low=low, ci95_high=high)


class ExperimentRecorder:
    """把实验结果落到 ``<root>/<exp_id>/``：

    - ``episodes.jsonl`` 逐行追加（崩溃也不丢已完成的 episode）
    - ``meta.json``      实验配置快照（保证结果可复现）
    - ``summary.json``   汇总（成功率 ± SE、失败模式分布、耗时）
    """

    def __init__(self, root: str | Path, exp_id: str, meta: dict[str, Any] | None = None):
        self.dir = Path(root) / exp_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.exp_id = exp_id
        self.episodes_path = self.dir / "episodes.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.summary_path = self.dir / "summary.json"
        if meta is not None:
            self.meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def log_episode(self, result: EpisodeResult) -> None:
        if result.exp_id != self.exp_id:
            raise ValueError(f"exp_id mismatch: {result.exp_id!r} != {self.exp_id!r}")
        with self.episodes_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(result.to_json() + "\n")

    def load_episodes(self) -> list[EpisodeResult]:
        if not self.episodes_path.exists():
            return []
        out: list[EpisodeResult] = []
        for line in self.episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(EpisodeResult(**json.loads(line)))
        return out

    def write_summary(self) -> dict[str, Any]:
        episodes = self.load_episodes()
        success_flags = [e.success for e in episodes]
        sr = summarize(success_flags)
        failures: dict[str, int] = {}
        for e in episodes:
            failures[e.failure_mode] = failures.get(e.failure_mode, 0) + 1
        durations = [e.duration_s for e in episodes]
        summary: dict[str, Any] = {
            "exp_id": self.exp_id,
            "num_episodes": sr.n,
            "num_successes": sr.successes,
            "success_rate": sr.rate,
            "success_rate_se": sr.se,
            "success_rate_ci95": [sr.ci95_low, sr.ci95_high],
            "failure_modes": failures,
            "duration_s_mean": statistics.fmean(durations) if durations else float("nan"),
            "duration_s_median": statistics.median(durations) if durations else float("nan"),
            "steps_mean": statistics.fmean([e.steps for e in episodes]) if episodes else float("nan"),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary


def _selftest() -> int:
    """不依赖任何外部库/仿真器的自查。"""
    with tempfile.TemporaryDirectory() as tmp:
        rec = ExperimentRecorder(tmp, "selftest-exp", meta={"note": "selftest"})
        # 构造 40 条：30 成功、10 失败，失败模式可区分
        for i in range(40):
            ok = i < 30
            rec.log_episode(
                EpisodeResult(
                    exp_id="selftest-exp",
                    task="PickCube-v1",
                    policy="dummy",
                    seed=i % 3,
                    episode_idx=i,
                    success=ok,
                    steps=50 + i,
                    duration_s=1.0 + 0.01 * i,
                    failure_mode="success" if ok else ("timeout" if i % 2 else "no_contact"),
                )
            )
        summary = rec.write_summary()
        assert summary["num_episodes"] == 40, summary
        assert summary["num_successes"] == 30, summary
        assert abs(summary["success_rate"] - 0.75) < 1e-12, summary
        # 校验 JSONL 可回读且条数一致
        assert len(rec.load_episodes()) == 40
        # 校验 Wilson 区间包含点估计且落在 [0,1]
        lo, hi = summary["success_rate_ci95"]
        assert 0.0 <= lo <= 0.75 <= hi <= 1.0, summary
        # 校验失败模式统计
        assert summary["failure_modes"]["success"] == 30
        assert sum(summary["failure_modes"].values()) == 40
        # 空输入必须显式返回 nan，而不是伪装成 0
        empty = summarize([])
        assert math.isnan(empty.rate) and empty.n == 0
        # 非法 failure_mode 必须报错
        try:
            EpisodeResult("e", "t", "p", 0, 0, True, 1, 0.1, failure_mode="made_up")
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid failure_mode was accepted")
        print("[selftest] "
              f"rate={summary['success_rate']:.3f} "
              f"se={summary['success_rate_se']:.4f} "
              f"ci95=[{lo:.3f}, {hi:.3f}] "
              f"failures={summary['failure_modes']}")
        print("[selftest] OK")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="embodied_core.metrics self-test / summary tool")
    parser.add_argument("--selftest", action="store_true", help="run built-in self-test")
    parser.add_argument("--root", default="experiments", help="experiment root dir")
    parser.add_argument("--exp-id", help="print summary for this exp_id")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.selftest:
        return _selftest()
    if args.exp_id:
        rec = ExperimentRecorder(args.root, args.exp_id)
        print(json.dumps(rec.write_summary(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
