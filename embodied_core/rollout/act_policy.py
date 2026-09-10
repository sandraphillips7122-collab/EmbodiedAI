"""ACT 策略适配器（EmbodiedCore · rollout）。

把 ManiSkill 官方 ACT baseline 的 checkpoint 接入 EmbodiedCore 的统一 ``Policy`` 接口，
让项目 A 的训练产物能被**同一个评测器**（``embodied_core.rollout.runner``）评测
—— 这是项目 A 与项目 B 结果可直接对照的前提。

复用的官方实现（``third_party/ManiSkill-main/examples/baselines/act``）：

- 模型：``act.detr.{backbone,transformer,detr_vae}`` 构建的 ``DETRVAE``
- obs 处理：``train_rgbd.FlattenRGBDObservationWrapper`` 的等价逻辑
  （每相机 rgb → resize 224 → ``(num_cams, 3, 224, 224)``；其余状态展平成 ``state``）
- 推理细节：``act/evaluate.py`` 的 **temporal aggregation**（时间集成）

**零侵入设计**：模型维度从 checkpoint 的权重形状推断、obs 预处理在内部完成，
因此不需要给 env 套额外 wrapper，也不需要改 ``adapter`` / ``runner``。

两点必须与训练保持一致（均已按官方语义实现）：

1. **delta 控制模式下不做归一化**：``train_rgbd.py`` 只在 ``delta_control`` 为假时才计算
   ``norm_stats``（第 233 行），checkpoint 里因此可能是 ``None``；官方 ``evaluate.py`` 用
   ``delta_control = not stats`` 表达了同一件事。本类在 ``norm_stats=None`` 时跳过
   state 标准化与 action 反归一化。
2. **temporal aggregation**：对最近 ``num_queries`` 次预测做指数加权平均（k=0.01），
   每步都要查询策略（``query_frequency=1``）。

用法::

    from embodied_core.rollout.act_policy import ActPolicy
    policy = ActPolicy(".../checkpoints/5000.pt", device="cuda", max_timesteps=200)
    summary = run_rollout(cfg, policy=policy)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = REPO_ROOT / "third_party" / "ManiSkill-main" / "examples" / "baselines" / "act"

#: 与官方 FlattenRGBDObservationWrapper / Agent 保持一致
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _import_act_modules():
    """导入官方 ACT baseline 的模型构建函数与 Args。"""
    if not ACT_DIR.exists():
        raise FileNotFoundError(
            f"未找到 ACT baseline 源码：{ACT_DIR}\n"
            "见 docs/env_decision.md 的下载源说明（gh-proxy.com 拉取 ManiSkill 源码）。"
        )
    if str(ACT_DIR) not in sys.path:
        sys.path.insert(0, str(ACT_DIR))

    import train_rgbd as act_train  # noqa: PLC0415
    from act.detr.backbone import build_backbone  # noqa: PLC0415
    from act.detr.detr_vae import DETRVAE, build_encoder  # noqa: PLC0415
    from act.detr.transformer import build_transformer  # noqa: PLC0415

    return act_train.Args, build_backbone, build_transformer, build_encoder, DETRVAE


class ActPolicy:
    """ACT（Action Chunking Transformer）策略，按 ``Policy`` 协议实现。"""

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        args_overrides: dict[str, Any] | None = None,
        device: str = "cuda",
        temporal_agg: bool = True,
        max_timesteps: int = 200,
    ) -> None:
        Args, build_backbone, build_transformer, build_encoder, DETRVAE = _import_act_modules()

        self.ckpt_path = Path(ckpt_path)
        self.args = Args(**{**(args_overrides or {}), "include_depth": False})
        self.device = torch.device(device)
        self.temporal_agg = bool(temporal_agg)
        self.max_timesteps = int(max_timesteps)
        self.name = f"act-ema:{self.ckpt_path.stem}"

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
        weights = ckpt.get("ema_agent") or ckpt["agent"]
        # delta 控制模式下官方不计算 norm stats（train_rgbd.py:233）→ 此时不做归一化
        self.stats = ckpt.get("norm_stats") or None

        # 维度从权重形状推断（既不依赖 env 的 observation_space，也不依赖 norm_stats）
        state_dim = int(weights["model.input_proj_robot_state.weight"].shape[1])
        action_dim = int(weights["model.action_head.weight"].shape[0])
        num_queries = int(weights["model.query_embed.weight"].shape[0])

        model = DETRVAE(
            [build_backbone(self.args)],
            build_transformer(self.args),
            build_encoder(self.args),
            state_dim=state_dim,
            action_dim=action_dim,
            num_queries=num_queries,
        )
        # checkpoint 存的是 Agent.state_dict()（键带 "model." 前缀），剥离前缀以匹配裸 DETRVAE
        model.load_state_dict({k.removeprefix("model."): v for k, v in weights.items()})
        self.model = model.to(self.device).eval()

        self.num_queries = num_queries
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._reset_buffers()

    # ------------------------------------------------------------------ 协议

    def reset(self, *, seed: int | None = None) -> None:
        self._reset_buffers()

    def predict(self, obs: Any, instruction: str | None = None) -> np.ndarray:
        """obs 为 ManiSkill 原始观测（dict，含 ``sensor_data`` / ``agent`` / ``extra``）。"""
        with torch.no_grad():
            tensor_obs = self._prepare_obs(obs)
            action_seq, _ = self.model(tensor_obs, None)  # (1, num_queries, action_dim)
            action = self._temporal_aggregate(action_seq)
            if self.stats is not None:
                action = action * self.stats["action_std"].to(self.device) + self.stats[
                    "action_mean"
                ].to(self.device)
        return action.to(torch.float32).cpu().numpy().reshape(self.action_dim)

    # ------------------------------------------------------------ obs 处理

    def _prepare_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        """复现官方 ``FlattenRGBDObservationWrapper.observation`` 的逻辑。

        ``CPUGymWrapper`` 会把观测转成 numpy 并去掉 batch 维；而官方 wrapper 是在
        ``CPUGymWrapper`` **内层**运行的，它看到的输入天然带 batch 维。因此这里先把
        观测树里的数值字段统一成「带 batch 维的 torch 张量」，再走官方的展平逻辑。
        """

        def batched(value):
            if isinstance(value, dict):
                return {k: batched(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(batched(v) for v in value)
            if isinstance(value, torch.Tensor):
                return value.unsqueeze(0) if value.dim() == 0 else value
            tensor = torch.as_tensor(np.asarray(value))
            return tensor.unsqueeze(0) if tensor.dim() <= 1 else tensor

        obs = batched(dict(obs))
        sensor_data = obs.pop("sensor_data", None)
        obs.pop("sensor_param", None)
        if not sensor_data:
            raise ValueError("观测中没有 sensor_data（需要 obs_mode 含 rgb）")

        # 每相机 rgb → resize 224；图像是 (H,W,C)，需先补出 batch 维
        images = []
        for cam in sensor_data.values():
            img = cam["rgb"]
            if img.dim() == 3:
                img = img.unsqueeze(0)
            images.append(self._resize(img.permute(0, 3, 1, 2)))  # (B,3,H,W)->(B,3,224,224)
        rgb = torch.stack(images, dim=1)  # (B, num_cams, 3, 224, 224), uint8

        # 其余字段（agent 的 qpos/qvel、extra 的 tcp_pose/goal_pos 等）展平成 state
        from mani_skill.utils import common

        state = common.flatten_state_dict(obs, use_torch=True)  # (B, state_dim)

        rgb = self._normalize(rgb.to(self.device).float() / 255.0)
        state = state.to(self.device).float()
        if self.stats is not None:
            state = (state - self.stats["state_mean"].to(self.device)) / self.stats[
                "state_std"
            ].to(self.device)
        return {"rgb": rgb, "state": state}

    # --------------------------------------------------- temporal aggregation

    def _reset_buffers(self) -> None:
        self._ts = 0
        self._all_time_actions: torch.Tensor | None = None

    def _temporal_aggregate(self, action_seq: torch.Tensor) -> torch.Tensor:
        """官方 ``evaluate.py`` 的时间集成：对最近 ``num_queries`` 次预测做指数加权平均。"""
        if not self.temporal_agg:
            raise NotImplementedError("当前只实现了 temporal_agg=True 的推理路径")

        horizon = self.max_timesteps + self.num_queries
        if self._all_time_actions is None:
            self._all_time_actions = torch.zeros(
                1, self.max_timesteps, horizon, self.action_dim, device=self.device
            )
        if self._ts >= self.max_timesteps:
            raise RuntimeError(
                f"超过 max_timesteps={self.max_timesteps}；"
                "请让 max_timesteps ≥ 评测的 max_episode_steps"
            )

        ts = self._ts
        self._all_time_actions[:, ts, ts : ts + self.num_queries] = action_seq
        actions_for_curr_step = self._all_time_actions[:, :, ts]  # (1, max_timesteps, act_dim)

        populated = torch.zeros(self.max_timesteps, dtype=torch.bool, device=self.device)
        populated[max(0, ts + 1 - self.num_queries) : ts + 1] = True
        actions_for_curr_step = actions_for_curr_step[:, populated]

        k = 0.01
        weights = torch.exp(
            -k
            * torch.arange(
                actions_for_curr_step.shape[1], device=self.device, dtype=torch.float32
            )
        )
        weights = (weights / weights.sum()).reshape(1, -1, 1)
        raw_action = (actions_for_curr_step * weights).sum(dim=1)  # (1, act_dim)

        self._ts += 1
        return raw_action[0]
