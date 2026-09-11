"""SmolVLA 策略适配器（EmbodiedCore · rollout）。

把 lerobot 的 SmolVLA 策略接入统一的 ``Policy`` 接口，使**项目 B 能被与项目 A 完全相同的
评测器**（``embodied_core.rollout.runner``）评测 —— 两个项目的结果因此可直接对照。

要点（均在实测中确认，见 `docs/s2_smolvla_smoke.md`）：

- **输入**：3 个相机各 `(B, 3, 256, 256)` + `observation.state`（我们 29 维；模型内部
  `pad_vector` 到 `max_state_dim=32`）+ 语言 token
- **推理**：`select_action(batch)` 自带 **action queue**（chunk 推理 + 逐步出队），
  每个环境步调用一次即可，不需要自己做 action chunking
- **动作**：模型输出 `max_action_dim=32` 维 → 我们只取前 `action_dim=8` 维
- **语言**：官方要求指令**以换行符结尾**（见 `processor_smolvla.py`），本类会自动补

用法::

    from embodied_core.rollout.smolvla_policy import SmolVLAPolicyAdapter
    policy = SmolVLAPolicyAdapter("<ckpt_dir>", instruction="pick up the red cube")
    summary = run_rollout(cfg, policy=policy)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

#: 与 DatasetPipe / replay 的相机命名一致
CAMERAS = ("camera1", "camera2", "camera3")
DEFAULT_INSTRUCTION = "pick up the red cube and move it to the goal position"


class SmolVLAPolicyAdapter:
    """SmolVLA（lerobot）→ EmbodiedCore ``Policy`` 协议。"""

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        device: str = "cuda",
        action_dim: int = 8,
    ) -> None:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.ckpt_path = str(ckpt_path)
        self.device = torch.device(device)
        self.instruction = instruction if instruction.endswith("\n") else instruction + "\n"
        self.action_dim = int(action_dim)

        self.policy = SmolVLAPolicy.from_pretrained(self.ckpt_path).to(self.device).eval()
        # 官方 pre/post processor：特征重命名、归一化、语言 tokenize，以及**输出反归一化**。
        # 不做反归一化的话动作会明显越界（实测出现过 ~2.8 这种值，而动作空间是 [-1,1]）。
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=self.ckpt_path
        )
        self.name = f"smolvla:{Path(self.ckpt_path).name}"
        self.reset()

    # ------------------------------------------------------------------ 协议

    def reset(self, *, seed: int | None = None) -> None:
        """清空 action queue（每个 episode 开始时调用）。"""
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs: Any, instruction: str | None = None) -> np.ndarray:
        batch = self.preprocessor(self._build_obs(obs, instruction))
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.reshape(-1)[: self.action_dim].to(torch.float32).cpu().numpy()

    # ------------------------------------------------------------ 输入构造

    def _build_obs(self, obs: dict, instruction: str | None) -> dict[str, Any]:
        """构造符合 LeRobot 约定的**原始**观测（tokenize / 归一化交给官方 preprocessor）。"""
        text = instruction or self.instruction
        if not text.endswith("\n"):
            text += "\n"

        sensor_data = obs["sensor_data"]
        batch: dict[str, Any] = {"task": text}
        for cam in CAMERAS:
            if cam not in sensor_data:
                raise KeyError(
                    f"观测缺少相机 {cam!r}（实际只有 {list(sensor_data)}）。"
                    "SmolVLA 的 image_features 固定为 camera1/2/3："
                    "需用 scripts/replay_3cam.py 生成 3 相机数据，"
                    "并用 make_env(spec, cameras='three') 构建评测环境。"
                )
            img = torch.as_tensor(np.asarray(sensor_data[cam]["rgb"]))
            if img.ndim == 3:  # (H,W,C) -> (1,H,W,C)
                img = img.unsqueeze(0)
            batch[f"observation.images.{cam}"] = img.permute(0, 3, 1, 2).to(self.device).float()

        batch["observation.state"] = self._flatten_state(obs)
        return batch

    def _flatten_state(self, obs: dict) -> torch.Tensor:
        """与 ``DatasetPipe`` / ``ActPolicy`` 保持同一展平顺序，避免训练—推理不一致。"""
        agent, extra = obs["agent"], obs["extra"]
        parts = [
            np.asarray(agent["qpos"], dtype=np.float32).reshape(-1),
            np.asarray(agent["qvel"], dtype=np.float32).reshape(-1),
            np.asarray(extra["is_grasped"], dtype=np.float32).reshape(-1),
            np.asarray(extra["tcp_pose"], dtype=np.float32).reshape(-1),
            np.asarray(extra["goal_pos"], dtype=np.float32).reshape(-1),
        ]
        flat = np.concatenate(parts).reshape(1, -1)
        return torch.as_tensor(flat, dtype=torch.float32, device=self.device)
