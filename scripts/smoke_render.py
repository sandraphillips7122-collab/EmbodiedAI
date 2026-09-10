"""S0 冒烟测试 · SAPIEN 渲染设备与 RGB 输出（Windows 原生）

目的：确认 Windows 上 SAPIEN 能初始化 Vulkan 渲染设备（GPU），并能正确输出
RGB + Segmentation。**不依赖任何 ManiSkill 资产**，只用内置几何体。

用法::

    .venv\\Scripts\\python.exe scripts\\smoke_render.py --out experiments/smoke/sapien_render.png

判据：渲染设备为 GPU；RGB 有明暗层次（非全黑/全白）；目标立方体像素数 > 500。
退出码 0 表示通过。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import sapien
except Exception:  # pragma: no cover
    import traceback

    traceback.print_exc()
    raise SystemExit(1)


CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CUBE_HALF_SIZE = 0.1
CUBE_COLOR = [0.9, 0.2, 0.2, 1.0]
# 相机位于 (0, 0, 0.8)，绕 Y 轴旋转 90°，使其光轴由 +X 转到 -Z（俯视原点）
CAMERA_POSE = dict(p=[0.0, 0.0, 0.8], q=[0.70710678, 0.0, 0.70710678, 0.0])


def build_scene() -> "sapien.Scene":
    scene = sapien.Scene()
    scene.set_timestep(1.0 / 100.0)
    scene.add_ground(altitude=0.0)
    # 环境光是必须的：只加定向光时画面会全黑
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([1.0, 0.0, -1.0], [2.0, 2.0, 2.0], shadow=True)

    material = sapien.render.RenderMaterial(base_color=CUBE_COLOR)
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[CUBE_HALF_SIZE] * 3, material=material)
    builder.add_box_collision(half_size=[CUBE_HALF_SIZE] * 3)
    builder.build(name="red_cube")
    return scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/smoke/sapien_render.png")
    args = parser.parse_args()

    print("sapien:", sapien.__version__)
    print("render device summary:", sapien.render.get_device_summary().strip())
    print("camera shader dir:", sapien.render.get_camera_shader_dir())

    scene = build_scene()
    camera = scene.add_camera(
        name="cam", width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fovy=1.0, near=0.05, far=100.0
    )
    camera.set_pose(sapien.Pose(**CAMERA_POSE))

    for _ in range(3):
        scene.step()
        scene.update_render()
    camera.take_picture()

    color = np.asarray(camera.get_picture("Color"))[..., :3]
    rgb = (np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8) if color.dtype != np.uint8 else color
    segmentation = np.asarray(camera.get_picture("Segmentation"))
    seg_ids, seg_counts = np.unique(segmentation, return_counts=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(rgb).save(out_path)

    red_pixels = int(((rgb[..., 0].astype(int) > 120) & (rgb[..., 1].astype(int) < 120)).sum())
    unique_colors = len(np.unique(rgb.reshape(-1, 3), axis=0))
    print(f"rgb: min={int(rgb.min())} max={int(rgb.max())} mean={rgb.mean():.1f}")
    print(f"segmentation: {dict(zip(seg_ids.tolist(), seg_counts.tolist()))}")
    print(f"saved: {out_path}  unique_colors={unique_colors}  red_pixels={red_pixels}")

    passed = unique_colors > 50 and red_pixels > 500 and int(rgb.max()) > 150
    print("RENDER_OK" if passed else "RENDER_SUSPECT")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
