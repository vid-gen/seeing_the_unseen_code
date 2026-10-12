"""
DaS (DiffusionAsShader) 风格 Tracking Video 生成器

RGB 编码空间位置信息:
  R = 数据自适应归一化 X (第一帧, min/max)
  G = 数据自适应归一化 Y (第一帧, min/max)
  B = 归一化逆深度 1/Z (第一帧, 2%-98% percentile)

颜色固定自第一帧，点移动但颜色不变。
不可见/越界点不绘制 (黑色背景)。
"""

import os
import numpy as np
import cv2
from tqdm import tqdm


class DaSVisualizer:
    def __init__(self, save_dir="./results", fps=8, point_size=0, vis_threshold=0.9):
        """
        Args:
            save_dir: 输出目录
            fps: 输出帧率 (DaS 默认 8fps)
            point_size: 矩形半边长, 0=自适应 (基于最近邻距离)
            vis_threshold: 可见性阈值, 低于此值的点不渲染
        """
        self.save_dir = save_dir
        self.fps = fps
        self.point_size = point_size
        self.vis_threshold = vis_threshold

    def compute_colors(self, tracks_frame0, H, W):
        """从第一帧坐标计算 RGB 颜色编码。

        Args:
            tracks_frame0: (N, 3) → [x, y, depth]
            H, W: 视频分辨率

        Returns:
            colors: (N, 3) uint8 → [R, G, B]
        """
        N = tracks_frame0.shape[0]
        colors = np.zeros((N, 3), dtype=np.uint8)

        # R = data-adaptive normalized X (与 DaS SpaTracker 一致)
        x_vals = tracks_frame0[:, 0]
        x_min, x_max = x_vals.min(), x_vals.max()
        if x_max > x_min:
            colors[:, 0] = (np.clip((x_vals - x_min) / (x_max - x_min), 0, 1) * 255).astype(np.uint8)
        else:
            colors[:, 0] = 128

        # G = data-adaptive normalized Y (与 DaS SpaTracker 一致)
        y_vals = tracks_frame0[:, 1]
        y_min, y_max = y_vals.min(), y_vals.max()
        if y_max > y_min:
            colors[:, 1] = (np.clip((y_vals - y_min) / (y_max - y_min), 0, 1) * 255).astype(np.uint8)
        else:
            colors[:, 1] = 128
        # B = normalized inverse depth (2%-98% percentile)
        z_vals = tracks_frame0[:, 2]
        if np.all(z_vals == 0):
            colors[:, 2] = 128
        else:
            inv_z = 1.0 / (z_vals + 1e-10)
            p2, p98 = np.percentile(inv_z, 2), np.percentile(inv_z, 98)
            norm_z = np.clip((inv_z - p2) / (p98 - p2 + 1e-10), 0, 1)
            colors[:, 2] = (norm_z * 255).astype(np.uint8)

        return colors

    def sort_by_depth(self, depths):
        """按深度降序排列 (远→近, painter's algorithm)。"""
        return depths.argsort()[::-1]

    def compute_rect_size(self, tracks_frame0, vis_frame0=None):
        """计算自适应矩形半边长。

        基于第一帧中可见点的最小邻居距离。

        Args:
            tracks_frame0: (N, 2+) 第一帧坐标
            vis_frame0: (N,) 可见性, None 则全部视为可见

        Returns:
            rect_size: int, 矩形半边长
        """
        pts = tracks_frame0[:, :2]
        if vis_frame0 is not None:
            pts = pts[vis_frame0]

        if len(pts) < 2:
            return 4

        # 用所有可见点之间的距离估算
        # 为效率起见，随机采样一部分点
        n = min(len(pts), 500)
        idx = np.random.choice(len(pts), n, replace=False)
        sampled = pts[idx]

        min_dists = []
        for i in range(n):
            dists = np.linalg.norm(sampled - sampled[i], axis=1)
            dists = dists[dists > 0]
            if len(dists) > 0:
                min_dists.append(dists.min())

        if len(min_dists) == 0:
            return 4

        rect_size = int(np.median(min_dists) / 2)
        return max(rect_size, 1)

    def render_frame(self, tracks_t, vis_t, colors, rect_size, H, W):
        """渲染单帧。

        Args:
            tracks_t: (N, 3) 当前帧 [x, y, depth]
            vis_t: (N,) 可见性
            colors: (N, 3) uint8 颜色
            rect_size: int 矩形半边长
            H, W: 分辨率

        Returns:
            frame: (H, W, 3) uint8
        """
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # 先筛选画面内的点，再按 vis 阈值过滤
        pixels_all = tracks_t[:, :2].astype(int)
        in_frame = (pixels_all[:, 0] >= 0) & (pixels_all[:, 0] < W) & \
                   (pixels_all[:, 1] >= 0) & (pixels_all[:, 1] < H)
        vis_mask = vis_t > self.vis_threshold
        keep = in_frame & vis_mask

        if not np.any(keep):
            return frame

        pixels = pixels_all[keep]
        depths = tracks_t[keep, 2]
        frame_colors = colors[keep]

        if len(pixels) == 0:
            return frame

        # 深度排序 (远→近)
        sort_idx = self.sort_by_depth(depths)
        pixels = pixels[sort_idx]
        frame_colors = frame_colors[sort_idx]

        # 画填充矩形
        for j in range(len(pixels)):
            x, y = int(pixels[j, 0]), int(pixels[j, 1])
            top_left = (x - rect_size, y - rect_size)
            bottom_right = (x + rect_size, y + rect_size)
            cv2.rectangle(frame, top_left, bottom_right,
                          frame_colors[j].tolist(), thickness=-1)

        return frame

    def save_video(self, frames, filename):
        """保存帧序列为 H.264 MP4。"""
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"{filename}.mp4")
        tmp_path = save_path + ".tmp.mp4"

        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_path, fourcc, self.fps, (W, H))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # 用 ffmpeg 重编码为 H.264
        ret = os.system(
            f'ffmpeg -y -i "{tmp_path}" -c:v libx264 -pix_fmt yuv420p '
            f'"{save_path}" -loglevel warning'
        )
        if ret == 0 and os.path.exists(save_path):
            os.remove(tmp_path)
        else:
            # ffmpeg 不可用时保留 mp4v 版本
            if os.path.exists(tmp_path):
                os.rename(tmp_path, save_path)

        print(f"DaS tracking video saved to {save_path}")

    def generate(self, tracks_3d, visibility, H, W, filename="tracking_das"):
        """生成 DaS 风格 tracking video。

        Args:
            tracks_3d: (T, N, 3) → [x, y, depth] 每帧每点
            visibility: (T, N) 或 (T, N, 1) → 可见性
            H, W: 输出分辨率
            filename: 输出文件名 (不含扩展名)
        """
        # 处理 visibility 维度
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]

        T, N, _ = tracks_3d.shape
        print(f"[DaSVisualizer] Generating: {T} frames, {N} points, {W}x{H}")

        # 1) 计算第一帧颜色
        colors = self.compute_colors(tracks_3d[0], H, W)

        # 2) 计算矩形大小
        if self.point_size > 0:
            rect_size = self.point_size
        else:
            rect_size = self.compute_rect_size(
                tracks_3d[0], vis_frame0=(visibility[0] > self.vis_threshold)
            )
        print(f"[DaSVisualizer] Rect size: {rect_size}")

        # 3) 逐帧渲染
        frames = []
        for t in tqdm(range(T), desc="Rendering DaS tracking frames"):
            frame = self.render_frame(
                tracks_3d[t], visibility[t], colors, rect_size, H, W
            )
            frames.append(frame)

        # 4) 保存视频
        self.save_video(frames, filename)
        print(f"[DaSVisualizer] Done. Frames: {T}, Resolution: {W}x{H}, "
              f"Points: {N}, Rect size: {rect_size}")
