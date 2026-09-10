# 环境决策记录（Phase 1 · S0）

探测与验证日期：2026-09-10（本机系统时间）

## 1. 主机事实（实测）

| 项 | 实测值 | 证据 |
|---|---|---|
| 主机 | Windows + RTX 4060 Laptop（8188 MiB 显存） | `nvidia-smi`：driver 615.65.07 / CUDA UMD 13.4 |
| **PyTorch 可用显存** | **7106 MiB free / 8188 MiB total**（Windows 桌面占约 1.1 GB） | `torch.cuda.mem_get_info()` |
| PyTorch | `torch 2.9.1+cu128`，`cuda_available=True`，capability **(8, 9)**，CUDA 12.8 | venv 内实测 |
| CPU / 内存（WSL 视角） | 32 逻辑核 / 15 GiB | `nproc`, `free -h` |
| WSL 发行版 | Ubuntu 24.04.4 LTS，kernel 6.6.87.2-microsoft-standard-WSL2，systemd=true | `uname -a` |
| WSL interop | 可用（可从 WSL 调 Windows 可执行文件） | `cmd.exe /c echo INTEROP_OK` → `INTEROP_OK` |
| WSL 的 Python | `/usr/bin/python3` = 3.12.3，但 **`python3 -m venv` 失败**（缺 `python3.12-venv`，需 sudo 密码） | `ensurepip is not available` |
| Windows 的 Python | `D:\python\python.exe` = 3.13.2；`C:\Users\29446\anaconda3\python.exe` = 3.12.7 | `python.exe --version` |
| 磁盘余量 | C: 96 GiB 可用；E: 108 GiB 可用 | `df -h` |
| sudo | 需要密码（无法自动安装 WSL 系统包） | `sudo -n true` |

## 2. 关键能力矩阵（SAPIEN 官方能力）

| 系统 / GPU | CPU Sim | GPU Sim | Rendering |
|---|---|---|---|
| Linux / NVIDIA | ✅ | ✅ | ✅ |
| **Windows / NVIDIA** | ✅ | ❌ | ✅（但见 §8 约束 1） |
| **WSL / 任意** | ✅ | ❌ | **❌** |

## 3. WSL 渲染不可用的实测证据

- `/etc/vulkan/icd.d/` 为空；`/usr/share/vulkan/icd.d/` 只有 Mesa 系，**没有 `nvidia_icd.json`**。
- `/usr/lib/x86_64-linux-gnu/` 下没有 `libvulkan_dzn.so`（Vulkan→D3D12 翻译层）。
- 结论：WSL 内 Vulkan 只能落到 `llvmpipe`（CPU 软渲染），与官方矩阵一致。**WSL 不能作为运行环境。**

## 4. 决策

**执行侧：Windows 原生。**（Python 环境 `E:\EmbodiedAI\.venv`，Python 3.12.7，由 conda base 的 python 创建、与 anaconda 完全隔离：`include-system-site-packages = false`。）

WSL 仅作为发起命令的 shell 通道（通过 interop 调用 Windows 侧程序）。

## 5. Python 版本与 wheel 兼容性

- `mani_skill 3.0.1`（`py3-none-any`）按平台分派依赖：Linux → `sapien>=3.0.0`；**Windows → `sapien>=3.0.0b1`**；macOS → `sapien>=3.0.2`。
- `sapien 3.0.3` 提供 `win_amd64` wheel（cp310–cp314）→ **cp312 可用**。
- `lerobot` 要求 `requires_python >= 3.12`。
- 统一锁定 **Python 3.12**。

## 6. 依赖下载源选型（实测速度）

| 源 | 实测 | 结论 |
|---|---|---|
| 清华 PyPI 镜像 | 0.55 s | ✅ 纯 Python 包 |
| 官方 `pypi.org` / `download.pytorch.org` | 超时 / 9.2 KB/s | ❌ |
| 上海交大 `pytorch-wheels` | 29 KB/s | ❌ |
| **阿里云 `mirrors.aliyun.com/pytorch-wheels`** | **14–16 MB/s** | ✅ CUDA torch 专用源 |
| GitHub 直连 | 超时 | ❌ |
| **`gh-proxy.com`** | **4.4–17 MB/s** | ✅ GitHub 源码/归档 |
| `hf-mirror.com` | 1.4–8.7 MB/s | ✅ HuggingFace 数据 |

### 陷阱（务必记住）

1. **清华 PyPI 的 `torch` win wheel 是 CPU-only**（124 MB，`cuda_available=False`）。必须用带 `+cuXXX` 的 wheel（约 2.5 GB）。
2. 阿里云 `pytorch-wheels` 是**文件级镜像**（无 PEP 503 索引）→ 需直接下载 wheel 再本地安装。
3. 版本配对：`torch 2.9.1+cu128` ↔ `torchvision 0.24.1+cu128`（cp312/win_amd64）。
4. 大体积 wheel 放工作区外临时目录（`%LOCALAPPDATA%\Temp\ea_wheels`），不入库。

## 7. S0 验证结果（全部为实测）

| 判据 | 结果 | 证据 |
|---|---|---|
| PyTorch CUDA | ✅ | `torch 2.9.1+cu128`，`cuda_available=True`，可用显存 7106 MiB |
| SAPIEN 渲染设备 | ✅ **GPU：NVIDIA GeForce RTX 4060 Laptop GPU**（cudaId 0, rayTrace 1） | `scripts/smoke_render.py` |
| SAPIEN RGB + Segmentation | ✅ `RENDER_OK`，`red_pixels=5476` | `experiments/smoke/sapien_render.png` |
| ManiSkill3 安装 | ✅ `mani_skill 3.0.1` + `sapien 3.0.3` | `pip list` |
| env 构建（state） | ✅ | `scripts/smoke_env.py` |
| 示教下载/解压 | ✅ 36.6 MB zip → `~/.maniskill/demos/PickCube-v1`（90 MB） | 目录列表 |
| PickCube 额外资产 | ✅ **无需下载**（TableSceneBuilder + `build_cube` 程序化物体 + 包内 panda） | `pick_cube.py` + 包内 `assets/robots/panda` 159 MB |
| **rgb 示教 replay** | ✅ `Replayed 4 episodes, 4/4=100.00%`，**86–200 step/s** | `trajectory.rgb.pd_joint_pos.physx_cpu.h5`（1.59 MB） |

## 8. Windows 平台的两个硬约束（关键，必须遵守）

### 约束 1：ManiSkill 默认 `render_backend="gpu"` 在本机会崩溃

- 现象：`obs_mode='rgb'` 构建 env 时进程**原生崩溃**（exit 5，无 Python traceback，输出因块缓冲丢失）。
- 根因：env 的 `render_backend` 默认 `"gpu"`（`sapien_cuda`）。
- 解法：显式设 `render_backend="cpu"`（`sapien_cpu`）→ 实测构建成功、渲染正常。
- 代价：渲染走 CPU。实测 replay（含多相机 RGB 保存）**86–200 step/s**，对本项目规模足够。
- 落地：示教 json 的 `env_info.env_kwargs` 补 `"render_backend": "cpu"`（`replay_trajectory.py` 会继承该字段）；原始文件已备份为 `trajectory.json.orig`。

### 约束 2：Windows 无 pinocchio → 末端笛卡尔控制（`pd_ee_*`）不可用

- 现象：`TypeError: 'NoneType' object is not callable`，抛在 `mani_skill/agents/controllers/utils/kinematics.py:115` → `PinocchioModel(xml, ...)`。
- 根因：`sapien/wrapper/pinocchio_model.py` **只在 `platform.system() == "Linux"` 时回退到内置的 `pysapien_pinocchio`**；其他平台将 `PinocchioModel = None`（仅打一条 warning）。
- 影响：`pd_ee_delta_pos` / `pd_ee_delta_pose` 等需要 IK 的控制模式**无法构建**；而 ManiSkill 官方 replay 脚本与 ACT/DP baseline 默认都用 `pd_ee_delta_pos`。
- 可行缓解：
  1. **关节空间控制**（`pd_joint_pos` / `pd_joint_delta_pos`）——不需要 IK，**已实测 replay 通过**（零额外下载）；
  2. conda-forge 的 `pinocchio` 提供 **win-64** 包（PyPI 的 `pin` 只有 manylinux/macOS）→ 需把环境重建为 conda 环境；
  3. Ubuntu 双系统（同时解决 IK 与 GPU 渲染）。

## 9. 待验证项

- [ ] Windows + `physx_cpu` 的 CPU 并行吞吐（多进程 `AsyncVectorEnv`，8 进程）
- [ ] ACT 训练真实峰值显存与单步耗时（→ `docs/vram_report.md`）
- [ ] `obs_mode='rgb'` + `render_backend='cpu'` 下**并行 rollout 评测**的吞吐
