"""导入自检：确认 ``embodied_core`` 各子模块在 pytest 下都能被导入。

这条测试存在的意义：本仓库曾出现「从仓库根运行脚本找不到 embodied_core」的问题
（``docs/phase3_progress.md`` 第 11 条）。这里用一个显式用例把该前提固定下来 ——
一旦导入路径再次坏掉，``pytest`` 会立刻失败，而不是等到某天跑脚本时才炸。
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "embodied_core",
    "embodied_core.envs.adapter",
    "embodied_core.envs.cameras",
    "embodied_core.metrics.recorder",
    "embodied_core.metrics.failure_modes",
    "embodied_core.rollout.runner",
]

# 这些模块的顶层 import 依赖 torch / lerobot / mani_skill，缺失时跳过而不是失败
OPTIONAL_MODULES = [
    "embodied_core.rollout.act_policy",
    "embodied_core.rollout.smolvla_policy",
    "embodied_core.data.pipeline",
]


@pytest.mark.parametrize("name", MODULES)
def test_core_module_imports(name: str) -> None:
    """这些模块只依赖标准库或已装好的轻量依赖，必须能导入。"""
    module = importlib.import_module(name)
    assert module is not None


@pytest.mark.parametrize("name", OPTIONAL_MODULES)
def test_optional_module_imports(name: str) -> None:
    """重依赖模块：装了对应依赖就必须能导入，没装则跳过（不算失败）。"""
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:  # torch / lerobot / mani_skill 未安装
        pytest.skip(f"{name} 依赖未安装：{exc.name}")
    assert module is not None
