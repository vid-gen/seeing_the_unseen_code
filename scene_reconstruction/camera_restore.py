#!/usr/bin/env python3
"""
camera_restore.py - 摄像头位置还原模块

从 MoGe 相机内参和 Physion 坐标变换参数，还原拍摄图片时的摄像头位置。
支持将 Physion 仿真空间中的 3D 点投影到原始图像平面。

坐标系链路:
  MoGe 相机空间 (OpenCV: X-right, Y-down, Z-forward)
    ↓ moge_to_scene: (-x, -y, z)  [翻转 X 和 Y]
  scene_posed.ply 空间 (X-left, Y-up, Z-forward, Z 增大=远离相机)
    ↓ align_to_physion_coords: 平移 + 缩放
  Physion 仿真空间 (Y-up, 地面 Y=0, XZ 居中)

用法:
  camera = CameraInPhysion.from_files(
      "output/camera_params.json",
      "output/trial/coord_transform.json"
  )
  # 投影 3D 点到 2D
  uv = camera.project(points_3d)
  # 获取相机在 Physion 空间的位置
  pos = camera.position
"""

import json
import math
import numpy as np
from typing import Optional, Tuple


class CameraInPhysion:
    """
    Physion 仿真空间中的还原相机

    Attributes:
        position: 相机在 Physion 空间的位置 (3,)
        look_direction: 相机朝向 (3,), 单位向量
        up: 相机 up 方向 (3,), 单位向量
        fov_x_deg: 水平 FOV (度)
        fov_y_deg: 垂直 FOV (度)
        fx_norm: 归一化焦距 x
        fy_norm: 归一化焦距 y
        cx_norm: 归一化主点 x
        cy_norm: 归一化主点 y
        image_width: 原始图像宽度 (像素)
        image_height: 原始图像高度 (像素)
    """

    def __init__(
        self,
        camera_params: dict,
        coord_transform: dict,
    ):
        """
        Args:
            camera_params: auto_scene_pipeline.py 输出的 camera_params.json
            coord_transform: gs_to_particles.py 输出的坐标变换参数
        """
        # 解析 intrinsics
        intrinsics = camera_params["intrinsics"]
        self.fx_norm = float(intrinsics[0][0])
        self.fy_norm = float(intrinsics[1][1])
        self.cx_norm = float(intrinsics[0][2])
        self.cy_norm = float(intrinsics[1][2])

        self.image_width = int(camera_params["image_width"])
        self.image_height = int(camera_params["image_height"])
        self.fov_x_deg = float(camera_params["fov_x_deg"])
        self.fov_y_deg = float(camera_params["fov_y_deg"])

        # 坐标变换参数
        self.ground_y = float(coord_transform["ground_y"])
        self.center_x = float(coord_transform["center_x"])
        self.center_z = float(coord_transform["center_z"])
        self.scale = float(coord_transform["scale"])

        # 场景缩放因子 (MoGe v2 米制输出 → Physion 训练分布范围)
        # 旧数据没有此字段，默认 1.0（不缩放）
        self.scene_scale_factor = float(
            coord_transform.get("scene_scale_factor", 1.0))

        # 计算相机在 Physion 空间的位置
        # 相机在 scene_posed 空间的位置: (0, 0, 0)
        # 完整变换: p_physion = (p_scene * scene_scale_factor - [center_x, ground_y, center_z]) * scale
        # 相机位置: 0 * ssf = 0，所以 scene_scale_factor 不影响相机位置
        # center_x/ground_y/center_z 已在缩放后的空间计算
        self.position = np.array([
            (0.0 - self.center_x) * self.scale,
            (0.0 - self.ground_y) * self.scale,
            (0.0 - self.center_z) * self.scale,
        ], dtype=np.float64)

        # 相机朝向和 up（scene_posed: 朝 +Z 看, Y-up）
        # 平移和缩放不改变方向
        self.look_direction = np.array([0.0, 0.0, 1.0])
        self.up = np.array([0.0, 1.0, 0.0])

        # 预计算相机旋转矩阵 (世界→相机)
        # 相机坐标系: X-right, Y-up, Z-forward (看向 +Z)
        # 世界坐标系: 与相机坐标系相同方向（只有平移差异）
        # 所以旋转矩阵 = I（单位矩阵）
        self._R_world_to_cam = np.eye(3)

        # scene_posed 到 MoGe (OpenCV) 的转换: (-x, -y, z)
        # scene_posed: X-left, Y-up, Z-forward (PyTorch3D camera_to_pytorch3d_camera 翻转了 X)
        # MoGe/OpenCV: X-right, Y-down, Z-forward
        # 需翻转 X 和 Y 轴: p_moge = (-p_cam_x, -p_cam_y, p_cam_z)
        self._R_cam_to_moge = np.diag([-1.0, -1.0, 1.0])

        # 地面偏移补偿量: 物体在构建 Physion 数据时被上移了 ground_clearance,
        # 投影时需要还原，使 2D 位置与原图一致
        # 旧数据兼容: 没有此字段时默认 0（旧数据的 ground_y 已包含补偿）
        self.ground_clearance = float(
            coord_transform.get("ground_clearance", 0.0))

        # 多米诺方向旋转逆矩阵：构建 trial 时场景被绕 Y 轴旋转了，
        # project() 中先逆旋转把 rollout 点转回原始空间，再用原始空间的 position 投影
        domino_R = coord_transform.get("domino_rotation_matrix", None)
        if domino_R is not None:
            R = np.array(domino_R, dtype=np.float64)
            self._domino_R_inv = R.T
        else:
            self._domino_R_inv = None

    @classmethod
    def from_files(
        cls,
        camera_params_path: str,
        coord_transform_path: str,
    ) -> "CameraInPhysion":
        """从 JSON 文件创建"""
        with open(camera_params_path, 'r') as f:
            camera_params = json.load(f)
        with open(coord_transform_path, 'r') as f:
            coord_transform = json.load(f)
        return cls(camera_params, coord_transform)

    @classmethod
    def from_scene_info(
        cls,
        camera_params_path: str,
        scene_info_path: str,
    ) -> "CameraInPhysion":
        """从 camera_params.json + scene_info.json (包含 coord_transform) 创建"""
        with open(camera_params_path, 'r') as f:
            camera_params = json.load(f)
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)
        coord_transform = scene_info["coord_transform"]
        return cls(camera_params, coord_transform)

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """
        将 Physion 世界坐标转换到相机坐标 (PyTorch3D 约定)

        Args:
            points: (N, 3) Physion 空间的 3D 点

        Returns:
            (N, 3) 相机坐标系中的点
        """
        # 平移到以相机为原点
        return (points - self.position) @ self._R_world_to_cam.T

    def project(
        self,
        points: np.ndarray,
        return_depth: bool = False,
    ) -> np.ndarray:
        """
        将 Physion 空间的 3D 点投影到 2D 图像平面

        投影流程:
        1. Physion → 相机坐标: 平移
        2. scene_posed → MoGe (OpenCV): (-x, -y, z) [翻转 X 和 Y]
        3. MoGe 投影: u = fx * X/Z + cx, v = fy * Y/Z + cy
        4. 归一化坐标 → 像素坐标

        Args:
            points: (N, 3) Physion 空间的 3D 点
            return_depth: 是否同时返回深度

        Returns:
            uv: (N, 2) 像素坐标 (u, v)
            如果 return_depth=True，返回 (uv, depth)
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, 3)

        # 0a. 逆旋转多米诺方向：从 Physion 旋转空间还原到原始 scene_posed 空间
        if self._domino_R_inv is not None:
            points = points @ self._domino_R_inv.T

        # 0b. 补偿 ground_clearance: 物体在 Physion 中被上移了 gc,
        #     投影时还原到原始高度，使 2D 位置与原图地面对齐
        if self.ground_clearance > 0:
            points = points.copy()
            points[:, 1] -= self.ground_clearance

        # 1. 世界 → 相机坐标 (PyTorch3D)
        p_cam = self.world_to_camera(points)

        # 2. scene_posed → MoGe (OpenCV): (-x, -y, z) [翻转 X 和 Y]
        p_moge = p_cam @ self._R_cam_to_moge.T

        # 3. 透视投影（MoGe 归一化坐标）
        z = p_moge[:, 2]
        # 避免除零（z <= 0 的点在相机后面）
        z_safe = np.where(np.abs(z) > 1e-8, z, 1e-8)

        u_norm = self.fx_norm * p_moge[:, 0] / z_safe + self.cx_norm
        v_norm = self.fy_norm * p_moge[:, 1] / z_safe + self.cy_norm

        # 4. 归一化 → 像素
        u_px = u_norm * self.image_width
        v_px = v_norm * self.image_height

        uv = np.stack([u_px, v_px], axis=1)

        if return_depth:
            return uv, z
        return uv

    def project_with_visibility(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        投影并过滤可见点

        Returns:
            uv: (M, 2) 可见点的像素坐标
            depth: (M,) 可见点的深度
            mask: (N,) 布尔 mask，True 表示可见
        """
        uv, depth = self.project(points, return_depth=True)

        # 可见性条件：深度 > 0 且在图像范围内
        visible = (
            (depth > 0) &
            (uv[:, 0] >= 0) & (uv[:, 0] < self.image_width) &
            (uv[:, 1] >= 0) & (uv[:, 1] < self.image_height)
        )

        return uv[visible], depth[visible], visible

    def get_extrinsics_4x4(self) -> np.ndarray:
        """
        获取 4x4 相机外参矩阵 (world → camera)

        Returns:
            (4, 4) 齐次变换矩阵
        """
        E = np.eye(4)
        E[:3, :3] = self._R_world_to_cam
        E[:3, 3] = -self._R_world_to_cam @ self.position
        return E

    def get_intrinsics_3x3_pixel(self) -> np.ndarray:
        """
        获取像素坐标的 3x3 内参矩阵

        Returns:
            K = [[fx_px, 0, cx_px], [0, fy_px, cy_px], [0, 0, 1]]
        """
        K = np.array([
            [self.fx_norm * self.image_width, 0, self.cx_norm * self.image_width],
            [0, self.fy_norm * self.image_height, self.cy_norm * self.image_height],
            [0, 0, 1],
        ])
        return K

    def compute_optical_flow(
        self,
        points_t0: np.ndarray,
        points_t1: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算两帧之间的稀疏光流

        Args:
            points_t0: (N, 3) 第0帧的 Physion 3D 点
            points_t1: (N, 3) 第1帧的 Physion 3D 点（同一批点，一一对应）
            image_size: (H, W) 输出图像大小，默认使用原图大小

        Returns:
            flow: (H, W, 2) 稠密光流图（稀疏点插值，未覆盖区域为 0）
            valid_mask: (H, W) 有效区域 mask
        """
        if image_size is None:
            image_size = (self.image_height, self.image_width)

        H, W = image_size

        # 投影两帧
        uv0, depth0 = self.project(points_t0, return_depth=True)
        uv1, depth1 = self.project(points_t1, return_depth=True)

        # 只保留两帧都可见的点
        visible = (
            (depth0 > 0) & (depth1 > 0) &
            (uv0[:, 0] >= 0) & (uv0[:, 0] < W) &
            (uv0[:, 1] >= 0) & (uv0[:, 1] < H)
        )

        uv0_vis = uv0[visible]
        uv1_vis = uv1[visible]

        # 计算 2D 位移
        displacement = uv1_vis - uv0_vis  # (M, 2)

        # 生成稀疏光流图
        flow = np.zeros((H, W, 2), dtype=np.float32)
        valid_mask = np.zeros((H, W), dtype=bool)

        for i in range(len(uv0_vis)):
            x, y = int(round(uv0_vis[i, 0])), int(round(uv0_vis[i, 1]))
            if 0 <= x < W and 0 <= y < H:
                flow[y, x] = displacement[i]
                valid_mask[y, x] = True

        return flow, valid_mask

    def render_trajectory_2d(
        self,
        trajectory: np.ndarray,
        instance_idx: np.ndarray,
        n_objects: int,
        image: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        colors: Optional[list] = None,
    ) -> np.ndarray:
        """
        将物体质心的 3D 轨迹投影到 2D 并绘制

        Args:
            trajectory: (n_steps, N, 3) 所有帧的粒子位置（Physion 空间）
            instance_idx: (n_objects+1,) 物体粒子索引边界
            n_objects: 物体数
            image: 背景图像 (H, W, 3)，如果为 None 则用白色背景
            output_path: 保存路径
            colors: 颜色列表

        Returns:
            canvas: 绘制后的图像 (H, W, 3)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if colors is None:
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                       '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

        H, W = self.image_height, self.image_width

        fig, ax = plt.subplots(1, 1, figsize=(W/100, H/100), dpi=100)

        # 背景
        if image is not None:
            ax.imshow(image)
        else:
            ax.set_facecolor('white')
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)  # 翻转 Y 轴

        n_steps = trajectory.shape[0]

        # 计算每个物体的质心轨迹
        for obj_id in range(n_objects):
            st = instance_idx[obj_id]
            ed = instance_idx[obj_id + 1]

            centroids_3d = []
            for t in range(n_steps):
                c = trajectory[t, st:ed].mean(axis=0)
                centroids_3d.append(c)
            centroids_3d = np.array(centroids_3d)  # (n_steps, 3)

            # 投影到 2D
            uv = self.project(centroids_3d)

            color = colors[obj_id % len(colors)]
            # 绘制轨迹线
            ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=2, alpha=0.8)
            # 起点和终点
            ax.scatter(uv[0, 0], uv[0, 1], color=color, s=80, zorder=5,
                       edgecolors='white', linewidths=1.5, marker='o')
            ax.scatter(uv[-1, 0], uv[-1, 1], color=color, s=80, zorder=5,
                       edgecolors='white', linewidths=1.5, marker='s')
            # 箭头
            if n_steps > 1:
                dx = uv[-1, 0] - uv[-2, 0]
                dy = uv[-1, 1] - uv[-2, 1]
                ax.annotate('', xy=(uv[-1, 0], uv[-1, 1]),
                            xytext=(uv[-1, 0] - dx*2, uv[-1, 1] - dy*2),
                            arrowprops=dict(arrowstyle='->', color=color, lw=2))

        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout(pad=0)

        if output_path:
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
            print(f"  Saved trajectory 2D: {output_path}")

        # 返回图像数组
        fig.canvas.draw()
        canvas = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        return canvas

    def to_dict(self) -> dict:
        """导出为 JSON 可序列化的 dict"""
        return {
            "position": self.position.tolist(),
            "look_direction": self.look_direction.tolist(),
            "up": self.up.tolist(),
            "fov_x_deg": self.fov_x_deg,
            "fov_y_deg": self.fov_y_deg,
            "intrinsics_norm": [
                [self.fx_norm, 0, self.cx_norm],
                [0, self.fy_norm, self.cy_norm],
                [0, 0, 1],
            ],
            "intrinsics_pixel": self.get_intrinsics_3x3_pixel().tolist(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coord_transform": {
                "ground_y": self.ground_y,
                "center_x": self.center_x,
                "center_z": self.center_z,
                "scale": self.scale,
                "ground_clearance": self.ground_clearance,
            },
        }

    def save(self, path: str):
        """保存相机参数到 JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Saved camera in Physion space: {path}")

    def __repr__(self):
        return (
            f"CameraInPhysion(\n"
            f"  position={self.position},\n"
            f"  look_dir={self.look_direction},\n"
            f"  fov={self.fov_x_deg:.1f}x{self.fov_y_deg:.1f} deg,\n"
            f"  image={self.image_width}x{self.image_height}\n"
            f")"
        )


def visualize_flow(flow: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """
    将光流转换为 HSV 色轮可视化

    Args:
        flow: (H, W, 2) 光流
        max_flow: 最大流量值（用于归一化），None 则自动计算

    Returns:
        vis: (H, W, 3) RGB 可视化图像 uint8
    """
    import cv2

    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx)

    if max_flow is None:
        max_flow = max(mag.max(), 1e-5)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue: 方向
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = np.clip(mag / max_flow * 255, 0, 255).astype(np.uint8)  # Value: 幅度

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def save_flo(flow: np.ndarray, path: str):
    """保存光流为 .flo 格式"""
    h, w = flow.shape[:2]
    with open(path, 'wb') as f:
        # Magic number
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([w, h], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='摄像头位置还原')
    parser.add_argument('--camera-params', type=str, required=True,
                        help='camera_params.json 路径')
    parser.add_argument('--coord-transform', type=str, default=None,
                        help='coord_transform.json 路径')
    parser.add_argument('--scene-info', type=str, default=None,
                        help='scene_info.json 路径 (包含 coord_transform)')
    parser.add_argument('--output', '-o', type=str, default='camera_physion.json',
                        help='输出路径')
    args = parser.parse_args()

    if args.coord_transform:
        camera = CameraInPhysion.from_files(args.camera_params, args.coord_transform)
    elif args.scene_info:
        camera = CameraInPhysion.from_scene_info(args.camera_params, args.scene_info)
    else:
        print("Error: 需要 --coord-transform 或 --scene-info")
        exit(1)

    print(camera)
    camera.save(args.output)
