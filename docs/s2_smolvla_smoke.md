# S2 · SmolVLA 显存 smoke test（项目 B）

实测日期：2026-09-11
硬件：NVIDIA GeForce RTX 4060 Laptop（8188 MiB），Windows 原生
环境：`lerobot 0.6.1` + `lerobot[smolvla]`（transformers 等）+ `torch 2.9.1+cu128`

> 方法：**先测再定**。不预设模型能否在 8 GB 上训练，用 forward + backward + 优化器 step 实测峰值显存，
> 再决定是否需要降级（LoRA / 冻结视觉编码器 / NF4 / 云 GPU）。

## 1. 模型规格（实测）

| 项 | 值 |
|---|---|
| 参数量 | **450.0 M**（其中**可训练 99.9 M**，默认已冻结一部分） |
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
- 计划中的降级阶梯（LoRA → 只训 action expert → NF4/QLoRA → 云 GPU）**暂不启用**，保留为扩容或换任务时的备选

## 4. 说明与局限

- 本测试用**随机张量**测显存上限；真实训练的数据加载不进显存，结论可迁移。
- 未启用 gradient checkpointing，也未做混合精度调参（当前配置已够用）。
- **尚未接入真实 LeRobotDataset**（Kep 的数据管道 `DatasetPipe` 是下一步）：需要把 ManiSkill replay 的
  `.h5` 转成 LeRobotDataset，并处理相机数量（我们只有 1 个相机，SmolVLA 期望 3 个）与语言指令字段。
- 相机数量差异是一个**已知待解决问题**：SmolVLA 的 `image_features` 定义了 `camera1/2/3`，
  而 PickCube 只有 `base_camera`。可行做法：给任务加 2 个额外视角相机（ManiSkill 支持多相机配置），
  或按官方做法复制/填充缺失相机并在文档中明确说明。

## 5. 复现

```cmd
cd /d E:\EmbodiedAI
.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lerobot "lerobot[smolvla]"
:: 权重（约 900 MB，走 hf-mirror）
.venv\Scripts\python.exe -c "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))"
```

随后运行本仓库的 S2 smoke 脚本（见 `docs/env_decision.md` 的依赖源说明）。
