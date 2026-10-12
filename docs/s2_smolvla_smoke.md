# S2 · SmolVLA 显存 smoke test（项目 B）

实测日期：2026-09-11
硬件：NVIDIA GeForce RTX 4060 Laptop（8188 MiB），Windows 原生
环境：`lerobot 0.6.1` + `lerobot[smolvla]`（transformers 等）+ `torch 2.9.1+cu128`

> 方法：**先测再定**。不预设模型能否在 8 GB 上训练，用 forward + backward + 优化器 step 实测峰值显存，
> 再决定是否需要降级（LoRA / 冻结视觉编码器 / NF4 / 云 GPU）。

## 1. 模型规格（实测）

| 项 | 值 |
|---|---|
| 参数量 | **450.0 M**（其中**可训练 99.88 M = 22.19%**；即官方默认的"冻结 VLM + 只训 action expert"，实测明细见 [`smolvla_audit.md`](smolvla_audit.md)） |
| `n_obs_steps` / `chunk_size` / `n_action_steps` | 1 / 50 / 50 |
| `max_state_dim` / `max_action_dim` | 32 / 32 |
| 图像输入 | **3 个相机**，各 `(3, 256, 256)` |
| VLM 骨干 | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`（层数裁剪至 16） |
| 语言 token 最大长度 | 48（`pad_language_to="max_length"`） |
| 加载后显存 | **883 MiB** |

## 2. 显存扫描（forward + backward + AdamW.step，随机张量）

| batch size | loss | **峰值显存** | 单步耗时 |
|---|---|---|---|
| 1 | 1.152 | **1722 MiB** | 1.11 s |
| 2 | 1.090 | **1897 MiB** | 0.33 s |
| 4 | 2.003 | **2490 MiB** | 0.37 s |
| 8 | 1.867 | **3654 MiB** | 0.58 s |

峰值含 Windows 桌面基线约 1.8 GB；GPU 总显存 8188 MiB。

## 3. 结论

**RTX 4060 8 GB 可以直接训练 SmolVLA，无需降级。**

- batch=8 峰值 3654 MiB → 剩余约 **4.5 GB** 余量
- 训练吞吐 0.33–0.58 s/step（batch 8）→ 10000 steps 约 1–1.6 小时
- 计划中的降级阶梯（LoRA → NF4/QLoRA → 云 GPU）**暂不启用**，保留为扩容或换任务时的备选
- ⚠️ **一处需要澄清的表述**：本项目实际训练时，"冻结视觉编码器 + 只训 action expert" 是 **SmolVLA 的官方默认配置**（`freeze_vision_encoder=true` / `train_expert_only=true`），
  **不是我们额外加的降级手段**。因此不要说成"未启用冻结"——准确说法是"采用官方默认的部分微调"。
  实测参数量明细见 [`smolvla_audit.md`](smolvla_audit.md)

## 4. 说明与局限

- 本测试用**随机张量**测显存上限；真实训练的数据加载不进显存，结论可迁移。
- 未启用 gradient checkpointing，也未做混合精度调参（当前配置已够用）。
- **已接入真实 LeRobotDataset**（后续已完成）：`embodied_core/data/pipeline.py` 把 ManiSkill replay 的
  `.h5` 转成 LeRobotDataset（3 相机 + 29 维 state + 8 维 action + **语言指令**）。
  实测规模：100 eps → 166 MB；500 eps（图片模式）→ 11 GB。
- **相机数量差异已解决**：SmolVLA 的 `image_features` 要求 `camera1/2/3`，而 PickCube 只有 `base_camera`。
  做法是给任务加 **3 个真实视角**（`embodied_core/envs/cameras.py`，数据生成与评测两侧共用），
  并用 `scripts/replay_3cam.py` 重新生成数据 —— 实测 1000/1000 replay 成功（5.05 GB）。
  **注意**：`_default_sensor_configs` 是 **property**，必须用 `property(...)` 覆盖。

## 5. 复现

```cmd
cd /d <你的仓库根目录>
.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lerobot "lerobot[smolvla]"
:: 权重（约 900 MB，走 hf-mirror）
.venv\Scripts\python.exe -c "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))"
```

随后运行本仓库的 S2 smoke 脚本（见 `docs/env_decision.md` 的依赖源说明）。
