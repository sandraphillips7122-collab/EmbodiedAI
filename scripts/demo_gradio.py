"""项目 B 的可交互 Demo（Gradio）：自然语言指令 → 机器人执行 → 实时状态反馈。

设计目标（对应计划 Phase 4）：把「语言条件 VLA 的闭环」做成看得见的演示，并**如实展示状态**：

- 输入：自然语言指令（可自由改写，用于观察指令泛化）+ 策略选择（SmolVLA / ACT 对照）+ 随机种子
- 输出：
  - 三个视角的相机画面（camera1/2/3，与训练时一致）
  - 实时状态面板：当前步、夹爪开合、是否已成功、单步延迟（ms）、累计成功率
- 说明：本 Demo 展示的是**仿真环境中的闭环链路完整性**；SmolVLA 在该数据规模下成功率很低
  （见 `reports/B.md`），ACT 基线约 0.39。Demo 不承诺成功率。

用法::

    .venv\\Scripts\\python.exe scripts\\demo_gradio.py --policy smolvla
    .venv\\Scripts\\python.exe scripts\\demo_gradio.py --policy act
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# 允许从任意工作目录运行：Python 只把「脚本所在目录」加入 sys.path，
# 因此在仓库根执行 `python scripts/demo_gradio.py` 时会找不到 embodied_core 包。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SMOLVLA_CKPT = REPO_ROOT / "experiments/B_smolvla_finetune/checkpoints/005000/pretrained_model"
ACT_CKPT = (
    REPO_ROOT
    / "third_party/ManiSkill-main/examples/baselines/act/runs/act-pickcube-rgb-100demos/checkpoints/25000.pt"
)


def _load_policy(kind: str, instruction: str):
    """加载策略（两种策略都接入统一 Policy 接口，便于对照）。"""
    if kind == "act":
        from embodied_core.rollout.act_policy import ActPolicy

        return ActPolicy(str(ACT_CKPT), device="cuda", max_timesteps=200), 8.0
    from embodied_core.rollout.smolvla_policy import SmolVLAPolicyAdapter

    return SmolVLAPolicyAdapter(str(SMOLVLA_CKPT), instruction=instruction, device="cuda"), 0.0


def _make_env():
    from embodied_core.envs.adapter import EnvSpec, make_env
    from embodied_core.envs.cameras import patch_pickcube_with_three_cameras

    patch_pickcube_with_three_cameras()
    return make_env(EnvSpec(max_episode_steps=150), cameras="three")


def build_app(default_policy: str, default_steps: int):
    import gradio as gr

    with gr.Blocks(title="EmbodiedAI · 语言条件操作 Demo") as app:
        gr.Markdown(
            "## 语言条件机器人操作 Demo（ManiSkill3 · 仿真）\n"
            "输入一句自然语言指令，策略在仿真里执行并实时回传状态。\n"
            "> 模型：SmolVLA（450M，微调 5000 steps）或 ACT（任务专用，25000 steps）。"
            "**仿真环境，无真机验证。**"
        )
        with gr.Row():
            instruction = gr.Textbox(
                value="pick up the red cube and move it to the goal position",
                label="自然语言指令（可自由改写，观察指令泛化）",
                lines=2,
                scale=4,
            )
        with gr.Row():
            policy_kind = gr.Dropdown(
                choices=["smolvla", "act"], value=default_policy, label="策略"
            )
            seed = gr.Number(value=0, precision=0, label="随机种子")
            max_steps = gr.Slider(20, 150, value=default_steps, step=10, label="最大步数")
        run_btn = gr.Button("执行", variant="primary")
        with gr.Row():
            cam1 = gr.Image(label="camera1（正视）", height=200)
            cam2 = gr.Image(label="camera2（侧视）", height=200)
            cam3 = gr.Image(label="camera3（俯视）", height=200)
        status = gr.JSON(label="实时状态")

        def run(instruction_text, kind, seed_value, steps):
            env = _make_env()
            try:
                policy, _ = _load_policy(kind, instruction_text)
                obs, _ = env.reset(seed=int(seed_value))
                policy.reset(seed=int(seed_value))

                history: list[bool] = []
                for step in range(int(steps)):
                    t0 = time.perf_counter()
                    action = policy.predict(obs, instruction_text)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    obs, _r, terminated, truncated, info = env.step(action)
                    grasp_closed = bool(np.asarray(obs["extra"]["is_grasped"]).reshape(-1)[0])
                    panels = [
                        np.asarray(obs["sensor_data"][c]["rgb"]).reshape(256, 256, 3)
                        for c in ("camera1", "camera2", "camera3")
                    ]
                    yield (
                        panels[0],
                        panels[1],
                        panels[2],
                        {
                            "step": step,
                            "is_grasped": grasp_closed,
                            "success": bool(info.get("success", False)),
                            "latency_ms": round(latency_ms, 2),
                            "cum_steps": step + 1,
                            "note": "成功条件：方块到达目标位（goal_thresh=0.025m）",
                        },
                    )
                    if bool(info.get("success", False)):
                        break
                    if terminated or truncated:
                        break
            finally:
                env.close()

        run_btn.click(
            run,
            inputs=[instruction, policy_kind, seed, max_steps],
            outputs=[cam1, cam2, cam3, status],
        )
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="语言条件操作 Demo")
    parser.add_argument("--policy", default="smolvla", choices=["smolvla", "act"])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    app = build_app(args.policy, args.steps)
    app.launch(share=args.share, inbrowser=False, server_name="127.0.0.1", server_port=7860)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
