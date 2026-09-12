"""pytest 全局配置。

两件事：

1. **导入路径兜底** —— 让 pytest 在任意工作目录下都能 import ``embodied_core``。
   （``pyproject.toml`` 的 ``pythonpath = ["."]`` 已覆盖常规情况，这里是双保险，
   因为本仓库历史上踩过「脚本所在目录 ≠ 仓库根」的坑，见 ``docs/phase3_progress.md`` 第 11 条。）
2. **``--runslow`` 开关** —— 端到端测试（需要仿真器与 GPU）默认跳过，
   只在显式加 ``--runslow`` 时才运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="运行 slow 测试（需要仿真器/GPU 的端到端测试）",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="默认跳过；加 --runslow 运行端到端测试")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """仓库根目录（测试里读 experiments/ 等用）。"""
    return REPO_ROOT


@pytest.fixture(scope="session")
def experiments_dir(repo_root: Path) -> Path:
    """实验产物目录。"""
    return repo_root / "experiments"
