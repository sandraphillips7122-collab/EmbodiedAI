# 数据卡（ManiSkill 示教）

- 数据文件：`C:\Users\<user>\.maniskill\demos\PickCube-v1\motionplanning\trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5`
- 轨迹总数：**1000**（本卡片统计前 100 条）
- episode 长度：min=49 / median=79 / mean=77.3 / max=100
- 抽样成功率：1.0
- 动作维度：**8**

## 动作统计（全量抽样）

| 维度 | min | max | mean | std |
|---|---|---|---|---|
| 0 | -0.5037000179290771 | 0.478300005197525 | 0.004800000227987766 | 0.09619999676942825 |
| 1 | -1.0 | 0.7408999800682068 | 0.002400000113993883 | 0.30149999260902405 |
| 2 | -0.27160000801086426 | 0.2662000060081482 | -0.0038999998942017555 | 0.04919999837875366 |
| 3 | -1.0 | 0.9968000054359436 | 0.014499999582767487 | 0.2599000036716461 |
| 4 | -0.21400000154972076 | 0.23549999296665192 | 0.0010000000474974513 | 0.048700001090765 |
| 5 | -0.9404000043869019 | 0.6712999939918518 | -0.008700000122189522 | 0.2680000066757202 |
| 6 | -1.0 | 1.0 | 0.0203000009059906 | 0.2535000145435333 |
| 7 | -1.0 | 1.0 | 0.07800000160932541 | 0.996999979019165 |

## h5 结构（第一条轨迹）

```json
{
  "obs": "{agent: {qpos: Dataset(75, 9) float32, qvel: Dataset(75, 9) float32}, extra: {is_grasped: Dataset(75,) bool, tcp_pose: Dataset(75, 7) float32, goal_pos: Dataset(75, 3) float32}, sensor_param: {camera1: {extrinsic_cv: Dataset(75, 3, 4) float32, cam2world_gl: Dataset(75, 4, 4) float32, intrinsic_cv: Dataset(75, 3, 3) float32}, camera2: {extrinsic_cv: Dataset(75, 3, 4) float32, cam2world_gl: Dataset(75, 4, 4) float32, intrinsic_cv: Dataset(75, 3, 3) float32}, camera3: {extrinsic_cv: Dataset(75, 3, 4) float32, cam2world_gl: Dataset(75, 4, 4) float32, intrinsic_cv: Dataset(75, 3, 3) float32}}, sensor_data: {camera1: {rgb: Dataset(75, 256, 256, 3) uint8}, camera2: {rgb: Dataset(75, 256, 256, 3) uint8}, camera3: {rgb: Dataset(75, 256, 256, 3) uint8}}}",
  "actions": "Dataset(74, 8) float32",
  "terminated": "Dataset(74,) bool",
  "truncated": "Dataset(74,) bool",
  "success": "Dataset(74,) bool",
  "env_states": "{actors: {table-workspace: Dataset(75, 13) float32, cube: Dataset(75, 13) float32, goal_site: Dataset(75, 13) float32}, articulations: {panda: Dataset(75, 31) float32}}"
}
```
