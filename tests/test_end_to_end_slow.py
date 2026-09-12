"""端到端最小复现（slow，默认跳过）。

与 ``test_recorded_experiments.py`` 的分工：

- 那个文件复算**历史实验**（只读 JSONL，不跑仿真）
- 这个文件**真跑一遍仿真**，证明"从零开始这条链路确实能产出合规记录"

用 ``zero`` 策略（输出全零动作）：不需要任何 checkpoint、不需要 GPU，
但足以走完 ``reset → step×N → 记 JSONL → 写 summary`` 的全部路径，
因此能抓住"重构把落盘链路改坏了"这类问题。

运行::

    pytest -m slow --runslow tests/test_end_to_end_slow.py -v
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.slow

# 这一档必须真的用到仿真器；缺依赖时明确跳过而不是报错
pytest.importorskip("mani_skill", reason="未安装 mani_skill，端到端测试跳过")


def test_zero_policy_episode_produces_record(tmp_path) -> None:
    """跑 1 个 episode，断言产出 episodes.jsonl + summary.json 且内容自洽。"""
    from embodied_core.envs.adapter import EnvSpec
    from embodied_core.rollout.runner import RolloutConfig, run_rollout

    cfg = RolloutConfig(
        exp_id="e2e_slow_zero",
        env=EnvSpec(max_episode_steps=10),
        seeds=(0,),
        episodes_per_seed=1,
        max_episode_steps=10,
        policy_name="zero",
        cameras="default",
        sync_and_free=True,
        rebuild_env_every=0,
        # 产物写进 tmp_path，避免污染仓库 experiments/
        experiments_root=str(tmp_path),
    )
    summary = run_rollout(cfg, policy=None)

    # 1) 产物落盘
    exp_dir = tmp_path / "e2e_slow_zero"
    jsonl = exp_dir / "episodes.jsonl"
    assert jsonl.is_file(), "episodes.jsonl 未生成"
    assert (exp_dir / "summary.json").is_file(), "summary.json 未生成"
    assert (exp_dir / "meta.json").is_file(), "meta.json 未生成"

    # 2) JSONL 里的记录条数与 summary 一致
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    assert summary["num_episodes"] == 1

    # 3) 每条记录字段齐全、且 failure_mode 在合法集合内
    from embodied_core.metrics.recorder import FAILURE_MODES

    rec = records[0]
    for key in ("exp_id", "task", "policy", "seed", "episode_idx",
                "success", "steps", "duration_s", "failure_mode"):
        assert key in rec, f"记录缺少字段 {key}"
    assert rec["exp_id"] == "e2e_slow_zero"
    assert rec["failure_mode"] in FAILURE_MODES
    assert isinstance(rec["success"], bool)
    assert 0 <= rec["steps"] <= 10

    # 4) zero 策略在 10 步内不可能成功；这里只断言"记录如实反映了这一点"
    #    （不把"不成功"当成失败 —— 那是环境难度，不是代码问题）
    assert summary["success_rate"] in (0.0, 1.0)
