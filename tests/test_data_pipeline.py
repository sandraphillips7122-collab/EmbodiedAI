"""``embodied_core.data.pipeline`` 的单元测试。

**重点验证 state 展平的维度与字段顺序** —— 这是训练侧与评测侧之间最容易出错、
而且出错后**极难察觉**的地方：

如果数据生成时把 state 拼成 ``qpos,qvel,...`` 而策略推理时按别的顺序读，
模型看到的数值就是"张冠李戴"的 —— 训练照样收敛、loss 照样下降，
但策略永远学不对，且现象会和"数据不够"一模一样，排查代价极高。

所以这里用**内存 h5py**（不依赖磁盘上的真实示教文件）逐段核对：
每一段是否被放在 (T, 29) 的正确列区间里。

真实示教文件若存在，额外做一次端到端一致性检查（不存在则跳过）。
"""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest

from embodied_core.data.pipeline import (
    CAMERAS,
    STATE_PARTS,
    build_features,
    build_state,
)

# 各段的宽度：qpos(9) + qvel(9) + is_grasped(1) + tcp_pose(7) + goal_pos(3) = 29
WIDTHS = {"qpos": 9, "qvel": 9, "is_grasped": 1, "tcp_pose": 7, "goal_pos": 3}
T = 7  # 步数用最小值即可，不需要真实 episode 长度

#: 真实示教文件（不入库；存在时才做端到端检查）
REAL_H5 = (
    Path.home()
    / ".maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
)


def _make_obs_group(f: h5py.File, t: int = T) -> h5py.Group:
    """构造一条假轨迹的 obs group，每段填入**可区分**的常数，便于定位列区间。"""
    obs = f.create_group("obs")
    agent = obs.create_group("agent")
    extra = obs.create_group("extra")

    # 每段一个独特常数：qpos=1.0, qvel=2.0, is_grasped=1, tcp_pose=4.0, goal_pos=5.0
    agent.create_dataset("qpos", data=np.full((t, WIDTHS["qpos"]), 1.0, dtype=np.float32))
    agent.create_dataset("qvel", data=np.full((t, WIDTHS["qvel"]), 2.0, dtype=np.float32))
    extra.create_dataset("is_grasped", data=np.ones((t,), dtype=bool))
    extra.create_dataset("tcp_pose", data=np.full((t, WIDTHS["tcp_pose"]), 4.0, dtype=np.float32))
    extra.create_dataset("goal_pos", data=np.full((t, WIDTHS["goal_pos"]), 5.0, dtype=np.float32))
    return obs


@pytest.fixture()
def obs_group() -> h5py.Group:
    """内存 h5 文件里的 obs group。"""
    with h5py.File(io.BytesIO(), "w") as f:
        yield _make_obs_group(f)


# --------------------------------------------------------------------------------------
# state 展平
# --------------------------------------------------------------------------------------

def test_state_shape_is_29(obs_group: h5py.Group) -> None:
    """总维度必须是 29（SmolVLA 的 max_state_dim=32 会自行 padding）。"""
    state = build_state(obs_group)
    assert state.shape == (T, 29)
    assert state.dtype == np.float32


def test_state_part_widths_sum_to_29() -> None:
    """把宽度写成常量，防止有人悄悄改了某段宽度却不更新文档。"""
    assert sum(WIDTHS.values()) == 29
    assert STATE_PARTS == ("qpos", "qvel", "is_grasped", "tcp_pose", "goal_pos")


def test_state_field_order_matches_declared_parts(obs_group: h5py.Group) -> None:
    """逐列核对字段顺序：每段的值必须落在它声明的列区间里。

    这条是**核心**：如果 build_state 里两段的顺序被调换，
    这里会因为"该是 1.0 的列出现了 2.0"而失败。
    """
    state = build_state(obs_group)
    expected = {"qpos": 1.0, "qvel": 2.0, "is_grasped": 1.0, "tcp_pose": 4.0, "goal_pos": 5.0}

    cursor = 0
    for part in STATE_PARTS:
        width = WIDTHS[part]
        block = state[:, cursor:cursor + width]
        assert np.all(block == expected[part]), (
            f"字段 {part} 的列区间 [{cursor},{cursor + width}) 数值不符；实际首行={state[0]}"
        )
        cursor += width
    assert cursor == 29


def test_is_grasped_is_expanded_to_one_column(obs_group: h5py.Group) -> None:
    """``is_grasped`` 在 h5 里是 (T,) 的一维 bool，必须 reshape 成 (T,1) 才能拼接。"""
    state = build_state(obs_group)
    # 位置：qpos(9) + qvel(9) = 18 起，宽度 1
    assert np.all(state[:, 18:19] == 1.0)


def test_state_respects_actual_timestep_count() -> None:
    """T 必须跟随输入，不能被写死。"""
    with h5py.File(io.BytesIO(), "w") as f:
        grid = _make_obs_group(f, t=3)
        assert build_state(grid).shape == (3, 29)


def test_state_first_and_grasped_columns_are_distinct(obs_group: h5py.Group) -> None:
    """反向检查：若实现把 qpos 与 tcp_pose 调换，第 0 列会变成 4.0。"""
    state = build_state(obs_group)
    assert state[0, 0] == 1.0, "第 0 列应属于 qpos"
    assert state[0, 18] == 1.0, "第 18 列应属于 is_grasped"


# --------------------------------------------------------------------------------------
# features 定义
# --------------------------------------------------------------------------------------

def test_features_include_all_three_cameras() -> None:
    features = build_features(state_dim=29, action_dim=8)
    for cam in CAMERAS:
        assert f"observation.images.{cam}" in features
        assert features[f"observation.images.{cam}"]["shape"] == (256, 256, 3)
    assert features["observation.state"]["shape"] == (29,)
    assert features["action"]["shape"] == (8,)


def test_features_dtype_follows_use_videos_flag() -> None:
    """``use_videos=False`` 时图像 dtype 必须是 ``image``。

    这是**真实踩过的坑**：之前硬编码 ``"video"``，lerobot 在
    ``LeRobotDataset.create(use_videos=False)`` 时直接抛错。
    """
    video_features = build_features(29, 8, use_videos=True)
    image_features = build_features(29, 8, use_videos=False)
    for cam in CAMERAS:
        assert video_features[f"observation.images.{cam}"]["dtype"] == "video"
        assert image_features[f"observation.images.{cam}"]["dtype"] == "image"


def test_features_default_is_video() -> None:
    """默认行为保持向后兼容（不传 use_videos 时按 video）。"""
    features = build_features(29, 8)
    assert features[f"observation.images.{CAMERAS[0]}"]["dtype"] == "video"


# --------------------------------------------------------------------------------------
# 真实示教文件的一致性检查（文件存在才跑）
# --------------------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_H5.is_file(), reason="真实示教 h5 不存在（不入库），跳过")
def test_real_demo_state_is_29_dims() -> None:
    """对真实示教做一次端到端检查：第 1 条轨迹展平后必须是 (T, 29)。"""
    with h5py.File(REAL_H5, "r") as f:
        traj_name = sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))[0]
        state = build_state(f[traj_name]["obs"])
    assert state.ndim == 2 and state.shape[1] == 29
    assert np.isfinite(state).all(), "state 里不应出现 nan/inf"
