"""embodied-core：项目 A / 项目 B 共享的基础设施。

分工（唯一允许两个项目共享的地方，详见 README）：

- ``envs``    : EnvAdapter — ManiSkill 任务封装、观测 schema、泛化扰动接口、成功判定
- ``data``    : DatasetPipe — ManiSkill ``.h5`` 示教 → LeRobotDataset v3、key/维度对齐、数据卡
- ``rollout`` : RolloutRunner — 多进程 CPU 仿真评测、固定 seed、MP4 录制
- ``metrics`` : 成功率 ± 标准误、失败模式标签、JSONL 落盘（本包不依赖 ManiSkill/torch）
- ``config``  : 实验配置 schema 与实验 ID 规范

诚实性约定：本仓库所有指标必须能追溯到 ``experiments/<exp_id>/`` 下的原始记录。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
