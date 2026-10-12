import os
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms as T  # noqa: F401

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.models.utils import get_points_on_a_grid


# ============================================================
# 任务 4.2: 模型加载函数
# ============================================================
def load_models(track_mode="offline", vo_points=756, tracker="v2",
                v1_checkpoint=None, v1_seq_length=12):
    """
    统一模型加载入口
    Args:
        track_mode: "offline" 或 "online" (V2 专用)
        vo_points: 追踪点数量 (V2 专用)
        tracker: "v1", "v2", 或 "hybrid"
        v1_checkpoint: V1 模型权重路径 (None 则使用默认路径)
        v1_seq_length: V1 滑动窗口长度 (默认 12)
    返回:
        v1/v2 模式: (vggt_model, tracker_model) 二元组
        hybrid 模式: (vggt_model, v1_tracker, v2_tracker) 三元组
    """
    if tracker == "hybrid":
        return _load_models_hybrid(track_mode, vo_points, v1_checkpoint, v1_seq_length)
    elif tracker == "v1":
        return _load_models_v1(v1_checkpoint, v1_seq_length)
    else:
        return _load_models_v2(track_mode, vo_points)


def _load_models_v2(track_mode="offline", vo_points=756):
    """加载 V2: VGGT4Track + SpaTrack2"""
    # VGGT4Track 前端
    vggt_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt_model.eval()
    vggt_model = vggt_model.to("cuda")

    # SpaTrack2 追踪模型
    if track_mode == "offline":
        tracker_model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        tracker_model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    tracker_model.spatrack.track_num = vo_points
    tracker_model.eval()
    tracker_model.to("cuda")

    return vggt_model, tracker_model


def _load_models_v1(checkpoint_path=None, seq_length=12):
    """加载 V1: VGGT4Track + SpaTrackerV1Adapter"""
    raise NotImplementedError("v1/hybrid trackers are not included in this release; use --tracker v2")


def _load_models_hybrid(track_mode="offline", vo_points=756,
                        v1_checkpoint=None, v1_seq_length=12):
    """加载 VGGT + V1 + V2, 用于 hybrid 模式"""
    raise NotImplementedError("v1/hybrid trackers are not included in this release; use --tracker v2")


def resample_track_depth(track2d, depth_maps):
    """
    用共享深度图替换 track2d 中的深度通道, 确保 DaS 颜色一致性。
    追踪器输出的 track2d[:,:,2] 是模型内部预测的深度,
    此函数将其替换为共享深度源在对应像素位置的深度值。

    Args:
        track2d: (T, N, 3) ndarray — [x, y, depth_原始]
        depth_maps: (T, H, W) ndarray — 共享深度图 (如 VGGT depth)

    Returns:
        track2d_new: (T, N, 3) ndarray — [x, y, depth_共享]
    """
    T, N, _ = track2d.shape
    H, W = depth_maps.shape[1], depth_maps.shape[2]
    result = track2d.copy()

    for t in range(T):
        x_idx = np.clip(np.round(track2d[t, :, 0]).astype(int), 0, W - 1)
        y_idx = np.clip(np.round(track2d[t, :, 1]).astype(int), 0, H - 1)
        result[t, :, 2] = depth_maps[t, y_idx, x_idx]

    return result


# ============================================================
# 任务 4.1: 视频加载函数
# ============================================================
def load_video_data(data_type, data_dir, video_name, fps, vggt_model):
    """
    根据 data_type 加载视频数据
    返回: (video_tensor, depth_tensor, intrs, extrs, unc_metric, orig_H, orig_W)
        video_tensor: (T, C, H, W) float tensor (可能经过 preprocess 缩放)
        depth_tensor: (T, H, W) numpy 或 tensor
        intrs: (T, 3, 3) numpy
        extrs: (T, 4, 4) numpy
        unc_metric: (T, H, W) numpy 或 None
        orig_H, orig_W: int — 原始视频分辨率 (缩放前)
    """
    import decord

    if data_type == "RGBD":
        npz_dir = os.path.join(data_dir, f"{video_name}.npz")
        data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
        video_tensor = torch.from_numpy(data_npz_load["video"] * 255)
        video_tensor = video_tensor[::fps]
        depth_tensor = data_npz_load["depths"][::fps]
        intrs = data_npz_load["intrinsics"][::fps]
        extrs = np.linalg.inv(data_npz_load["extrinsics"])[::fps]
        unc_metric = None
        orig_H, orig_W = video_tensor.shape[2], video_tensor.shape[3]

    elif data_type == "RGB":
        vid_dir = os.path.join(data_dir, f"{video_name}.mp4")
        video_reader = decord.VideoReader(vid_dir)
        video_tensor = torch.from_numpy(
            video_reader.get_batch(range(len(video_reader))).asnumpy()
        ).permute(0, 3, 1, 2)
        video_tensor = video_tensor[::fps].float()
        orig_H, orig_W = video_tensor.shape[2], video_tensor.shape[3]

        # VGGT4Track 前端估计深度和位姿
        video_tensor = preprocess_image(video_tensor)[None]
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = vggt_model(video_tensor.cuda() / 255)
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return video_tensor, depth_tensor, intrs, extrs, unc_metric, orig_H, orig_W


# ============================================================
# 任务 4.3: 覆盖检测函数
# ============================================================
def compute_coverage_map(track2d, vis_pred, H, W, vis_threshold=0.5, radius=5):
    """
    根据追踪结果，计算每帧的像素覆盖图 (用宽松阈值判断覆盖范围)
    输入:
        track2d: (T, N, 3) numpy — 2D追踪坐标 (x, y, depth)
        vis_pred: (T, N, 1) numpy — 可见性
        H, W: 帧尺寸
        vis_threshold: 覆盖判断用的宽松可见性阈值 (默认0.5)
        radius: 覆盖半径 (像素)
    返回:
        coverage_maps: (T, H, W) bool numpy 数组
    """
    T = track2d.shape[0]
    coverage_maps = np.zeros((T, H, W), dtype=bool)

    for t in range(T):
        vis_mask = vis_pred[t, :, 0] > vis_threshold
        coords = track2d[t, vis_mask, :2]

        canvas = np.zeros((H, W), dtype=np.uint8)
        for i in range(len(coords)):
            x, y = int(round(coords[i, 0])), int(round(coords[i, 1]))
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), radius, 1, -1)
        coverage_maps[t] = canvas > 0

    return coverage_maps


# ============================================================
# 几何投影: 将 frame 0 全像素 3D 点投影到帧 t (含 Z-Buffer 深度检查)
# ============================================================
def project_frame0_to_frame_t(point_map_0, c2w_0, c2w_t, intrs_t, H, W,
                               conf_depth_0=None, conf_threshold=0.5,
                               point_map_t=None, depth_margin=0.2):
    """
    将 frame 0 的所有像素 3D 点投影到帧 t 的像素空间，
    并通过 Z-Buffer 深度一致性检查过滤被遮挡的投影点。

    输入:
        point_map_0: (3, H, W) — frame 0 每像素的相机坐标系 3D 坐标
        c2w_0: (4, 4)          — frame 0 的 camera-to-world 矩阵
        c2w_t: (4, 4)          — 帧 t 的 camera-to-world 矩阵
        intrs_t: (3, 3)        — 帧 t 的相机内参矩阵
        H, W: int              — 帧尺寸
        conf_depth_0: (H, W) or None — frame 0 的深度置信度
        conf_threshold: float  — 深度置信度阈值
        point_map_t: (3, H, W) or None — 帧 t 每像素的相机坐标系 3D 坐标 (用于深度检查)
        depth_margin: float    — 深度一致性容差比例 (默认 0.2 = 20%)

    返回:
        projected_px: (K,) — 有效投影点的 x 像素坐标 (int)
        projected_py: (K,) — 有效投影点的 y 像素坐标 (int)
    """
    # 1. 提取 frame 0 相机坐标并过滤
    cam_pts = point_map_0.reshape(3, -1)  # (3, H*W)

    if conf_depth_0 is not None:
        valid_mask = (conf_depth_0.reshape(-1) >= conf_threshold) & (cam_pts[2] > 0)
    else:
        valid_mask = cam_pts[2] > 0

    cam_pts_valid = cam_pts[:, valid_mask]  # (3, M)

    # 2. 转换到世界坐标
    R_0 = c2w_0[:3, :3]
    t_0 = c2w_0[:3, 3:]
    world_pts = R_0 @ cam_pts_valid + t_0  # (3, M)

    # 3. 投影到帧 t 的相机坐标系
    w2c_t = np.linalg.inv(c2w_t)
    R_t = w2c_t[:3, :3]
    t_t = w2c_t[:3, 3:]
    cam_pts_t = R_t @ world_pts + t_t  # (3, M)

    # 4. 过滤: 在相机前方
    depth_mask = cam_pts_t[2, :] > 1e-6
    cam_pts_front = cam_pts_t[:, depth_mask]

    # 5. 投影到像素坐标
    pixel_coords = intrs_t @ cam_pts_front  # (3, K)
    px = pixel_coords[0] / pixel_coords[2]
    py = pixel_coords[1] / pixel_coords[2]
    projected_depths = cam_pts_front[2]  # 投影深度

    # 6. 过滤: 在画面范围内
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px_valid = px[in_bounds].astype(int)
    py_valid = py[in_bounds].astype(int)
    depths_valid = projected_depths[in_bounds]

    # 7. Z-Buffer 深度一致性检查
    if point_map_t is not None and len(px_valid) > 0:
        actual_depths = point_map_t[2, py_valid, px_valid]  # 帧 t 该像素的实际深度
        safe_actual = np.maximum(actual_depths, 1e-6)
        depth_ratio = depths_valid / safe_actual
        # 投影深度不应远大于实际深度 (投影点在可见表面后面)
        # 也不应远小于实际深度 (投影点对应的物体已经不在那里了)
        depth_consistent = ((depth_ratio < (1 + depth_margin)) &
                            (depth_ratio > (1 / (1 + depth_margin))))
        px_valid = px_valid[depth_consistent]
        py_valid = py_valid[depth_consistent]

    return px_valid, py_valid


# ============================================================
# 混合覆盖图: 几何投影 (静态场景) + Stage 1 追踪 (动态物体)
# ============================================================
def build_hybrid_coverage_map(point_map_0, c2w_0, c2w_t, intrs_t,
                               track2d_t, vis_t, H, W,
                               conf_depth_0=None, conf_threshold=0.5,
                               track_coverage_radius=5,
                               dilate_pixels=1,
                               point_map_t=None, depth_margin=0.2):
    """
    构建帧 t 的混合覆盖图:
      覆盖来源1: frame 0 全像素几何投影 + Z-Buffer 深度检查 (覆盖静态场景)
      覆盖来源2: Stage 1 追踪点实际位置 (覆盖动态物体)

    输入:
        point_map_0: (3, H, W)   — frame 0 每像素相机坐标系 3D 坐标
        c2w_0: (4, 4)            — frame 0 camera-to-world
        c2w_t: (4, 4)            — 帧 t camera-to-world
        intrs_t: (3, 3)          — 帧 t 内参
        track2d_t: (N, 3)        — Stage 1 追踪点在帧 t 的 [x, y, depth]
        vis_t: (N, 1)            — Stage 1 追踪点在帧 t 的可见性
        H, W: int                — 帧尺寸
        conf_depth_0: (H, W)     — frame 0 深度置信度
        conf_threshold: float    — 深度置信度阈值
        track_coverage_radius: int — Stage 1 追踪点的覆盖半径
        dilate_pixels: int       — 几何投影覆盖图的膨胀像素 (处理亚像素误差)
        point_map_t: (3, H, W) or None — 帧 t 每像素 3D 坐标 (用于 Z-Buffer 检查)
        depth_margin: float      — 深度一致性容差比例 (默认 0.2 = 20%)

    返回:
        coverage_map: (H, W) bool numpy 数组
    """
    coverage_map = np.zeros((H, W), dtype=np.uint8)

    # ---- 来源1: 几何投影覆盖 (静态场景, 含 Z-Buffer 深度检查) ----
    proj_px, proj_py = project_frame0_to_frame_t(
        point_map_0, c2w_0, c2w_t, intrs_t, H, W,
        conf_depth_0, conf_threshold,
        point_map_t=point_map_t, depth_margin=depth_margin
    )
    if len(proj_px) > 0:
        # 逐像素赋值 (投影点密度极高，不需要画圆)
        coverage_map[proj_py, proj_px] = 1
        # 轻微膨胀处理亚像素对齐误差
        if dilate_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate_pixels + 1, 2 * dilate_pixels + 1)
            )
            coverage_map = cv2.dilate(coverage_map, kernel, iterations=1)

    # ---- 来源2: Stage 1 追踪覆盖 (动态物体) ----
    vis_mask = vis_t[:, 0] > 0.5
    if vis_mask.sum() > 0:
        coords = track2d_t[vis_mask, :2]
        for i in range(len(coords)):
            x, y = int(round(coords[i, 0])), int(round(coords[i, 1]))
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(coverage_map, (x, y), track_coverage_radius, 1, -1)

    return coverage_map > 0


# ============================================================
# Future-Visible 查询点采样 (反向投影: frame t → frame 0)
# ============================================================
def sample_future_visible_queries_inverse(point_map, c2w_traj, intrs,
                                           grid_size, frame_H, frame_W,
                                           conf_depth=None,
                                           conf_threshold_t=0.3,
                                           depth_margin=0.3,
                                           track2d_s1=None,
                                           vis_pred_s1=None,
                                           stage1_filter_radius=5.0,
                                           stage1_vis_threshold=0.5):
    """
    使用反向投影检测真正的 future-visible 点:
    对 frame t 的每个网格点, 将其 3D 坐标投影回 frame 0,
    如果投影越界或被遮挡, 则该点是真正的 future-visible。

    输入:
        point_map: (T, 3, H, W) numpy — 每帧每像素的相机坐标系 3D 坐标
        c2w_traj: (T, 4, 4) numpy    — camera-to-world 变换矩阵
        intrs: (T, 3, 3) numpy       — 相机内参矩阵
        grid_size: int                — 网格密度
        frame_H, frame_W: int        — 帧尺寸
        conf_depth: (T, H, W) numpy or None — 深度置信度
        conf_threshold_t: float       — frame t 深度置信度阈值 (宽松, 仅过滤完全无效点)
        depth_margin: float           — 遮挡判断容差 (0.3 = 30%)
        track2d_s1: (T, N, 3) or None — Stage1 2D 追踪结果, 用于剔除动态物体的反投影误判 (方案 F')
        vis_pred_s1: (T, N, 1) or None — Stage1 可见性 (与 track2d_s1 配套)
        stage1_filter_radius: float   — 候选点距离 stage1 任一高质量追踪点 < 此半径 (像素) 则丢弃
        stage1_vis_threshold: float   — stage1 追踪点可见性阈值 (>0.5 算高质量)
    返回:
        queries: (M, 3) numpy array, 每行 [t, x, y], t > 0
    """
    T = point_map.shape[0]

    # 1. 生成网格点
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    grid_pts = grid_pts[0].numpy()  # (G, 2), [x, y]
    G = grid_pts.shape[0]
    grid_xs = np.round(grid_pts[:, 0]).astype(int).clip(0, frame_W - 1)
    grid_ys = np.round(grid_pts[:, 1]).astype(int).clip(0, frame_H - 1)

    # 2. 预计算 frame 0 的数据
    w2c_0 = np.linalg.inv(c2w_traj[0])  # (4, 4) world-to-camera for frame 0
    K_0 = intrs[0]                        # (3, 3) frame 0 intrinsics
    depth_0 = point_map[0, 2]             # (H, W) frame 0 depth

    # 3. 逐帧反向投影检测 (方案 H: 解耦 OOB 与 occluded)
    # 拆三个独立数组, 每帧检查所有「还没拿到 OOB」的格点
    # → OOB 检查得到二次机会, 不再被早期 occluded 误判抢先
    first_oob = np.full(G, -1, dtype=int)
    first_occ = np.full(G, -1, dtype=int)
    first_behind = np.full(G, -1, dtype=int)

    # ===== DIAG: 跟踪每个网格点的归属与各闸门通过率 =====
    ever_valid_depth = np.zeros(G, dtype=bool)
    ever_in_front = np.zeros(G, dtype=bool)
    diag_total_checks = 0
    diag_z_pass = 0
    diag_conf_pass = 0
    diag_silent_behind = 0
    diag_ratios = []

    for t in range(1, T):
        # 检查所有还没拿到 OOB 的格点 (即使已被标 occluded, 仍需继续找潜在 OOB)
        no_oob = first_oob == -1
        if no_oob.sum() == 0:
            break

        check_idx = np.where(no_oob)[0]
        xs = grid_xs[check_idx]
        ys = grid_ys[check_idx]
        K = len(xs)

        cam_x = point_map[t, 0, ys, xs]
        cam_y = point_map[t, 1, ys, xs]
        cam_z = point_map[t, 2, ys, xs]

        # ===== DIAG: 累计每帧通过率 =====
        diag_total_checks += K
        z_ok = cam_z > 1e-6
        diag_z_pass += int(z_ok.sum())
        if conf_depth is not None:
            conf_ok = conf_depth[t, ys, xs] >= conf_threshold_t
            diag_conf_pass += int(conf_ok.sum())
        else:
            diag_conf_pass += K

        valid_depth = cam_z > 1e-6
        if conf_depth is not None:
            valid_depth &= conf_depth[t, ys, xs] >= conf_threshold_t

        ever_valid_depth[check_idx[valid_depth]] = True

        valid_local = np.where(valid_depth)[0]
        if len(valid_local) == 0:
            continue

        cam_valid = np.stack([cam_x[valid_local],
                              cam_y[valid_local],
                              cam_z[valid_local]], axis=0)

        R_t = c2w_traj[t, :3, :3]
        t_t = c2w_traj[t, :3, 3:]
        world_pts = R_t @ cam_valid + t_t

        R_0 = w2c_0[:3, :3]
        t_0 = w2c_0[:3, 3:]
        cam_pts_0 = R_0 @ world_pts + t_0

        in_front = cam_pts_0[2] > 1e-6
        cam_front = cam_pts_0[:, in_front]
        front_local = valid_local[in_front]

        if len(front_local) == 0:
            # 全在 frame 0 相机后方 → behind (only if not set)
            behind_g = check_idx[valid_local]
            new_b = first_behind[behind_g] == -1
            first_behind[behind_g[new_b]] = t
            continue

        diag_silent_behind += int((~in_front).sum())
        ever_in_front[check_idx[valid_local[in_front]]] = True

        pixel_0 = K_0 @ cam_front
        px_0 = pixel_0[0] / pixel_0[2]
        py_0 = pixel_0[1] / pixel_0[2]
        proj_depth = cam_front[2]

        in_bounds = (px_0 >= 0) & (px_0 < frame_W) & (py_0 >= 0) & (py_0 < frame_H)

        # === OOB: check_idx 都是 first_oob=-1, 直接赋 t ===
        out_of_bounds = ~in_bounds
        if out_of_bounds.any():
            oob_g = check_idx[front_local[out_of_bounds]]
            first_oob[oob_g] = t

        # === Occluded: 只更新 first_occ == -1 的格点 ===
        in_bounds_idx = np.where(in_bounds)[0]
        if len(in_bounds_idx) > 0:
            px_int = px_0[in_bounds_idx].astype(int).clip(0, frame_W - 1)
            py_int = py_0[in_bounds_idx].astype(int).clip(0, frame_H - 1)
            actual_depth_0 = depth_0[py_int, px_int]

            proj_d = proj_depth[in_bounds_idx]
            safe_actual = np.maximum(actual_depth_0, 1e-6)
            occluded = proj_d > safe_actual * (1 + depth_margin)

            diag_ratios.append((proj_d - safe_actual) / safe_actual)

            if occluded.any():
                occ_g = check_idx[front_local[in_bounds_idx[occluded]]]
                new_o = first_occ[occ_g] == -1
                first_occ[occ_g[new_o]] = t

    # ===== 合并: OOB > occluded > behind =====
    first_future_visible = np.full(G, -1, dtype=int)
    reason = np.zeros(G, dtype=int)

    oob_mask = first_oob > 0
    first_future_visible[oob_mask] = first_oob[oob_mask]
    reason[oob_mask] = 1

    occ_only = (~oob_mask) & (first_occ > 0)
    first_future_visible[occ_only] = first_occ[occ_only]
    reason[occ_only] = 2

    behind_only = (~oob_mask) & (~occ_only) & (first_behind > 0)
    first_future_visible[behind_only] = first_behind[behind_only]
    reason[behind_only] = 3

    # 关键诊断: 多少 OOB 是被「方案 H」从 occluded 救出来的
    oob_saved_from_occ = int((oob_mask & (first_occ > 0) & (first_oob > first_occ)).sum())

    never_assigned = first_future_visible == -1
    diag_always_visible = int((never_assigned & ever_valid_depth & ever_in_front).sum())
    diag_partial_behind = int((never_assigned & ever_valid_depth & ~ever_in_front).sum())
    diag_depth_invalid = int((never_assigned & ~ever_valid_depth).sum())
    diag_n_oob = int((reason == 1).sum())
    diag_n_occ = int((reason == 2).sum())
    diag_n_behind_all = int((reason == 3).sum())
    diag_n_fv = diag_n_oob + diag_n_occ + diag_n_behind_all

    print(f"[inverse DIAG-H] grid G={G}  conf_threshold_t={conf_threshold_t}  depth_margin={depth_margin}  (decoupled OOB/occluded)")
    print(f"[inverse DIAG-H] future-visible total: {diag_n_fv}")
    print(f"[inverse DIAG-H]   ├─ out-of-bounds : {diag_n_oob}  (其中 {oob_saved_from_occ} 个是 H 方案救援: OOB 帧晚于 occluded 帧)")
    print(f"[inverse DIAG-H]   ├─ occluded      : {diag_n_occ}")
    print(f"[inverse DIAG-H]   └─ all behind-cam: {diag_n_behind_all}")
    print(f"[inverse DIAG-H] 「双重身份」格点数 (同时拿到 OOB 和 occluded): {int((oob_mask & (first_occ > 0)).sum())}")
    print(f"[inverse DIAG-H] not-future-visible total: {int(never_assigned.sum())}")
    print(f"[inverse DIAG-H]   ├─ always visible in frame 0           : {diag_always_visible}")
    print(f"[inverse DIAG-H]   ├─ depth-invalid in ALL frames         : {diag_depth_invalid}")
    print(f"[inverse DIAG-H]   └─ partial behind-cam (silently dropped): {diag_partial_behind}")
    print(f"[inverse DIAG-H] per-check pass rates  (累加 {diag_total_checks} 个 (grid,t) 检查, 比 H 之前增加~80%)")
    print(f"[inverse DIAG-H]   cam_z > 1e-6                  : {diag_z_pass}/{diag_total_checks} = {diag_z_pass/max(diag_total_checks,1):.3f}")
    print(f"[inverse DIAG-H]   conf_depth >= {conf_threshold_t:.2f}            : {diag_conf_pass}/{diag_total_checks} = {diag_conf_pass/max(diag_total_checks,1):.3f}")
    print(f"[inverse DIAG-H]   silent in_front=False (累计): {diag_silent_behind}")

    # ===== DIAG: 遮挡判断的 ratio 分布 + 阈值扫描 (治 A') =====
    if len(diag_ratios) > 0:
        all_r = np.concatenate(diag_ratios)
        N_r = len(all_r)
        signed_q = np.quantile(all_r, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        abs_q = np.quantile(np.abs(all_r), [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        print(f"[inverse DIAG-H] occlusion ratio (proj_d - actual_0)/actual_0  N={N_r}  (current depth_margin={depth_margin})")
        print(f"[inverse DIAG-H]   signed ratio quantiles  5%={signed_q[0]:+.3f}  10%={signed_q[1]:+.3f}  25%={signed_q[2]:+.3f}  50%={signed_q[3]:+.3f}  75%={signed_q[4]:+.3f}  90%={signed_q[5]:+.3f}  95%={signed_q[6]:+.3f}")
        print(f"[inverse DIAG-H]   abs ratio quantiles     5%={abs_q[0]:.3f}  10%={abs_q[1]:.3f}  25%={abs_q[2]:.3f}  50%={abs_q[3]:.3f}  75%={abs_q[4]:.3f}  90%={abs_q[5]:.3f}  95%={abs_q[6]:.3f}")
        print(f"[inverse DIAG-H]   threshold scan: # in-bounds checks that would be occluded under different margins")
        for m in [0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00]:
            n_occ = int((all_r > m).sum())
            print(f"[inverse DIAG-H]     depth_margin={m:.2f}: {n_occ}/{N_r} = {n_occ/N_r:.3f}")

    # 4. 构造 queries
    valid = first_future_visible > 0
    if valid.sum() == 0:
        print("[inverse] No future-visible points found.")
        return np.zeros((0, 3), dtype=np.float32)

    future_queries = np.column_stack([
        first_future_visible[valid].astype(np.float32),
        grid_pts[valid, 0],
        grid_pts[valid, 1],
    ])

    print(f"[inverse] {valid.sum()}/{G} grid points are future-visible "
          f"(frames {first_future_visible[valid].min()}-{first_future_visible[valid].max()})")

    # ===== 方案 F': stage1 覆盖反向验证 =====
    # 对每个候选 (t, x, y), 若 frame t 在 (x, y) 的 stage1_filter_radius 半径内
    # 存在任一高质量 stage1 追踪点 → 说明这个区域已被 stage1 追踪覆盖
    # (典型: 动态物体如摩托车手, 反向投影错判但 stage1 一直追着它) → 过滤掉
    if track2d_s1 is not None and vis_pred_s1 is not None and stage1_filter_radius > 0:
        n_before = len(future_queries)
        query_t_int = future_queries[:, 0].astype(int)
        query_xy = future_queries[:, 1:3]
        keep = np.ones(n_before, dtype=bool)
        radius2 = stage1_filter_radius ** 2

        # 按帧分组处理
        unique_t = np.unique(query_t_int)
        diag_dropped_per_frame = {}
        for tf in unique_t:
            mask_tf = query_t_int == tf
            cand_xy = query_xy[mask_tf]              # (Mt, 2)
            s1_vis_mask = vis_pred_s1[tf, :, 0] > stage1_vis_threshold
            if s1_vis_mask.sum() == 0:
                continue
            s1_xy = track2d_s1[tf, s1_vis_mask, :2]  # (Kt, 2)
            # 距离平方矩阵 (Mt, Kt)
            d2 = ((cand_xy[:, None, :] - s1_xy[None, :, :]) ** 2).sum(-1)
            min_d2 = d2.min(axis=1)
            drop_local = min_d2 < radius2
            global_idx = np.where(mask_tf)[0]
            keep[global_idx[drop_local]] = False
            n_drop_t = int(drop_local.sum())
            if n_drop_t > 0:
                diag_dropped_per_frame[int(tf)] = n_drop_t

        n_after = int(keep.sum())
        n_dropped = n_before - n_after
        print(f"[inverse F'] stage1-coverage filter: dropped {n_dropped}/{n_before} "
              f"(radius={stage1_filter_radius}px, vis_threshold={stage1_vis_threshold})")
        if len(diag_dropped_per_frame) > 0:
            top_frames = sorted(diag_dropped_per_frame.items(), key=lambda kv: -kv[1])[:10]
            print(f"[inverse F']   top frames by drop count: " +
                  ", ".join(f"t={t}:{n}" for t, n in top_frames))
        future_queries = future_queries[keep]
        if len(future_queries) > 0:
            kept_t = future_queries[:, 0].astype(int)
            print(f"[inverse F'] After F' filter: {len(future_queries)} queries, "
                  f"frames {kept_t.min()}-{kept_t.max()}")
        else:
            print(f"[inverse F'] WARNING: F' filter dropped ALL candidates!")

    return future_queries


# ============================================================
# Future-Visible 查询点采样 v2 (混合覆盖: 几何投影 + Z-Buffer + Stage1追踪) [旧版保留]
# ============================================================
def sample_future_visible_queries_v2(track2d_s1, vis_pred_s1,
                                      point_map, c2w_traj, intrs,
                                      grid_size, frame_H, frame_W,
                                      conf_depth=None,
                                      conf_threshold=0.5,
                                      track_coverage_radius=5,
                                      dilate_pixels=1,
                                      depth_margin=0.2):
    """
    使用混合覆盖方案 (几何投影 + Z-Buffer + Stage1追踪) 采样 future-visible 查询点。

    输入:
        track2d_s1: (T, N, 3) numpy  — Stage 1 的 2D 追踪结果
        vis_pred_s1: (T, N, 1) numpy — Stage 1 的可见性
        point_map: (T, 3, H, W) numpy — 每帧每像素的相机坐标系 3D 坐标
        c2w_traj: (T, 4, 4) numpy    — camera-to-world 变换矩阵
        intrs: (T, 3, 3) numpy       — 相机内参矩阵
        grid_size: int                — 网格密度
        frame_H, frame_W: int        — 帧尺寸
        conf_depth: (T, H, W) numpy or None — 深度置信度
        conf_threshold: float         — 深度置信度阈值
        track_coverage_radius: int    — Stage 1 追踪点覆盖半径
        dilate_pixels: int            — 几何投影覆盖膨胀像素
        depth_margin: float           — Z-Buffer 深度一致性容差 (默认 0.2 = 20%)
    返回:
        queries: (M, 3) numpy array, 每行 [t, x, y], t > 0
    """
    T = track2d_s1.shape[0]

    # 1. 生成和 stage1 相同的网格位置
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    grid_pts = grid_pts[0].numpy()  # (G, 2), [x, y]
    G = grid_pts.shape[0]

    # 2. 预计算 frame 0 的数据
    point_map_0 = point_map[0]           # (3, H, W)
    c2w_0 = c2w_traj[0]                  # (4, 4)
    conf_depth_0 = conf_depth[0] if conf_depth is not None else None

    # 3. 逐帧检查覆盖
    first_uncovered = np.full(G, -1, dtype=int)  # -1 表示一直被覆盖

    for t in range(1, T):  # 跳过 t=0
        # 还有未分配的网格点吗？
        unassigned_mask = first_uncovered == -1
        if unassigned_mask.sum() == 0:
            break  # 所有网格点都已找到首次未覆盖帧

        # 构建混合覆盖图 (含 Z-Buffer 深度检查)
        coverage_map = build_hybrid_coverage_map(
            point_map_0=point_map_0,
            c2w_0=c2w_0,
            c2w_t=c2w_traj[t],
            intrs_t=intrs[t],
            track2d_t=track2d_s1[t],
            vis_t=vis_pred_s1[t],
            H=frame_H, W=frame_W,
            conf_depth_0=conf_depth_0,
            conf_threshold=conf_threshold,
            track_coverage_radius=track_coverage_radius,
            dilate_pixels=dilate_pixels,
            point_map_t=point_map[t],
            depth_margin=depth_margin,
        )

        # 检查每个未分配的网格点是否被覆盖
        for g in range(G):
            if first_uncovered[g] != -1:
                continue
            gx, gy = int(round(grid_pts[g, 0])), int(round(grid_pts[g, 1]))
            if 0 <= gx < frame_W and 0 <= gy < frame_H:
                if not coverage_map[gy, gx]:
                    first_uncovered[g] = t
            else:
                first_uncovered[g] = t

    # 4. 构造 queries
    valid = first_uncovered > 0
    if valid.sum() == 0:
        print("[sample_v2] All grid points are always covered. No future-visible points.")
        return np.zeros((0, 3), dtype=np.float32)

    future_queries = np.column_stack([
        first_uncovered[valid].astype(np.float32),
        grid_pts[valid, 0],
        grid_pts[valid, 1],
    ])

    print(f"[sample_v2] {valid.sum()}/{G} grid points are future-visible "
          f"(appear in frames {first_uncovered[valid].min()}-{first_uncovered[valid].max()})")

    return future_queries


# ============================================================
# 任务 4.4: Future-Visible 查询点采样函数 (基于相同网格位置) [旧版保留]
# ============================================================
def sample_future_visible_queries(track2d_s1, vis_pred_s1, grid_size, frame_H, frame_W,
                                  coverage_vis_threshold=0.5, coverage_radius=5):
    """
    使用和 stage1 完全相同的网格位置，找出每个网格点最早"未被覆盖"的帧。
    覆盖判断使用宽松阈值 (默认0.5)，只判断"这个区域有没有被追踪到"。
    质量过滤由调用方在保存/可视化时单独处理。

    输入:
        track2d_s1: (T, N, 3) numpy — stage1 的 2D 追踪结果
        vis_pred_s1: (T, N, 1) numpy — stage1 的可见性
        grid_size: int — 和 stage1 相同的网格密度
        frame_H, frame_W: 帧尺寸
        coverage_vis_threshold: 覆盖判断用的宽松可见性阈值 (默认0.5)
        coverage_radius: 覆盖半径 (像素), 某个 stage1 点投影到网格点附近
                         多远以内算"覆盖了"该网格点
    返回:
        queries: (M, 3) numpy array, 每行 [t, x, y], t > 0
    """
    T = track2d_s1.shape[0]

    # 1. 生成和 stage1 完全相同的网格位置
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    grid_pts = grid_pts[0].numpy()  # (G, 2), 每行 [x, y]
    G = grid_pts.shape[0]

    # 2. 对每个网格点，找到它最早"未被覆盖"的帧
    #    "被覆盖" = 该帧存在某个 stage1 可见点距离该网格点 < coverage_radius
    first_uncovered = np.full(G, -1, dtype=int)  # -1 表示一直被覆盖

    for t in range(1, T):  # 跳过 t=0 (stage1 已经追踪了)
        vis_mask = vis_pred_s1[t, :, 0] > coverage_vis_threshold
        if vis_mask.sum() == 0:
            # 这帧没有任何可见点，所有未标记的网格点都在这帧首次未被覆盖
            unassigned = first_uncovered == -1
            first_uncovered[unassigned] = t
            continue

        visible_coords = track2d_s1[t, vis_mask, :2]  # (K, 2)

        for g in range(G):
            if first_uncovered[g] != -1:
                continue  # 已经找到了最早未覆盖帧

            gx, gy = grid_pts[g]
            # 计算该网格点到所有可见点的距离
            dists = np.sqrt((visible_coords[:, 0] - gx) ** 2 +
                            (visible_coords[:, 1] - gy) ** 2)
            if dists.min() > coverage_radius:
                first_uncovered[g] = t

    # 3. 构造 queries: 只取有效的 (first_uncovered != -1)
    valid = first_uncovered > 0
    if valid.sum() == 0:
        print("[sample] All grid points are always covered by stage 1. No future-visible points.")
        return np.zeros((0, 3), dtype=np.float32)

    future_queries = np.column_stack([
        first_uncovered[valid].astype(np.float32),
        grid_pts[valid, 0],
        grid_pts[valid, 1],
    ])

    print(f"[sample] {valid.sum()}/{G} grid points are future-visible "
          f"(appear in frames {first_uncovered[valid].min()}-{first_uncovered[valid].max()})")

    return future_queries


# ============================================================
# 任务 4.4.1: 质量过滤函数 (严格阈值)
# ============================================================
def filter_tracks_by_quality(track3d_pred, track2d_pred, vis_pred, conf_pred,
                             vis_threshold=0.7, conf_threshold=0.5):
    """
    用严格阈值过滤追踪点，只保留高质量的点 (在 N 维度上过滤)。
    一个点只要在任意帧上同时满足 vis > vis_threshold AND conf > conf_threshold，
    就保留该点在所有帧上的追踪结果。

    输入:
        track3d_pred: (T, N, D) — 3D 追踪坐标
        track2d_pred: (T, N, D2) — 2D 追踪坐标
        vis_pred: (T, N, 1) — 可见性
        conf_pred: (T, N, 1) — 置信度
        vis_threshold: 可见性严格阈值
        conf_threshold: 置信度严格阈值
    返回:
        filtered versions of (track3d_pred, track2d_pred, vis_pred, conf_pred), n_kept
    """
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    vis_np = to_np(vis_pred)
    conf_np = to_np(conf_pred)

    # 逐帧判断: 该点是否在至少一帧上同时满足两个严格阈值
    # shape: (T, N)
    quality_mask_per_frame = (vis_np[..., 0] > vis_threshold) & (conf_np[..., 0] > conf_threshold)
    # 在 N 维度上: 只要有任意一帧通过，就保留该点
    keep_mask = quality_mask_per_frame.any(axis=0)  # (N,)

    n_total = keep_mask.shape[0]
    n_kept = keep_mask.sum()
    print(f"[quality filter] Kept {n_kept}/{n_total} points "
          f"(vis>{vis_threshold} AND conf>{conf_threshold})")

    # ===== DIAG: 多角度通过率统计 =====
    # 每个点在所有帧上的最佳 vis / conf
    best_vis = vis_np[..., 0].max(axis=0)   # (N,)
    best_conf = conf_np[..., 0].max(axis=0) # (N,)

    print(f"[quality filter DIAG] N={n_total}  thresholds: vis>{vis_threshold} AND conf>{conf_threshold}")
    print(f"[quality filter DIAG] best_vis  per-point: min={best_vis.min():.3f} median={np.median(best_vis):.3f} max={best_vis.max():.3f}")
    print(f"[quality filter DIAG] best_conf per-point: min={best_conf.min():.3f} median={np.median(best_conf):.3f} max={best_conf.max():.3f}")

    # 仅 vis / 仅 conf / AND / OR 各自通过 (对当前阈值)
    vis_only_pass = int((best_vis > vis_threshold).sum())
    conf_only_pass = int((best_conf > conf_threshold).sum())
    or_pass = int(((vis_np[..., 0] > vis_threshold) | (conf_np[..., 0] > conf_threshold)).any(axis=0).sum())
    print(f"[quality filter DIAG] (any-frame) vis>{vis_threshold} alone   : {vis_only_pass}/{n_total}")
    print(f"[quality filter DIAG] (any-frame) conf>{conf_threshold} alone  : {conf_only_pass}/{n_total}")
    print(f"[quality filter DIAG] (any-frame) AND (current keep)         : {n_kept}/{n_total}")
    print(f"[quality filter DIAG] (any-frame) OR                          : {or_pass}/{n_total}")

    # 扫不同阈值下的 AND 通过数 (诊断阈值松紧影响)
    for thr in [0.3, 0.5, 0.7, 0.9]:
        m = ((vis_np[..., 0] > thr) & (conf_np[..., 0] > thr)).any(axis=0)
        print(f"[quality filter DIAG] (any-frame) vis>{thr} AND conf>{thr}: {int(m.sum())}/{n_total}")

    return (
        track3d_pred[:, keep_mask],
        track2d_pred[:, keep_mask],
        vis_pred[:, keep_mask],
        conf_pred[:, keep_mask],
        int(n_kept),
        keep_mask,
    )


# ============================================================
# 任务 4.5: 结果保存函数
# ============================================================
def save_tracking_results(output_dir, filename, c2w_traj, intrs, point_map,
                          conf_depth, track3d_pred, track2d_pred, vis_pred,
                          video_tensor):
    """
    保存 tapip3d 格式的 .npz 文件
    """
    os.makedirs(output_dir, exist_ok=True)

    # 确保都是 numpy
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    c2w_traj = to_np(c2w_traj)
    intrs = to_np(intrs)
    point_map = to_np(point_map)
    conf_depth = to_np(conf_depth)
    track3d_pred = to_np(track3d_pred)
    track2d_pred = to_np(track2d_pred)
    vis_pred = to_np(vis_pred)
    video_tensor = to_np(video_tensor)

    data = {}
    # coords: 世界坐标系下的 3D 轨迹
    c2w_rot = c2w_traj[:, :3, :3]  # (T, 3, 3)
    c2w_trans = c2w_traj[:, :3, 3]  # (T, 3)
    coords = np.einsum("tij,tnj->tni", c2w_rot, track3d_pred[:, :, :3]) + c2w_trans[:, None, :]
    data["coords"] = coords
    data["extrinsics"] = np.linalg.inv(c2w_traj)
    data["intrinsics"] = intrs
    depth_save = point_map[:, 2, ...] if point_map.ndim == 4 else point_map
    if conf_depth is not None:
        depth_save = depth_save.copy()
        depth_save[conf_depth < 0.5] = 0
    data["depths"] = depth_save
    data["video"] = video_tensor / 255.0 if video_tensor.max() > 1.0 else video_tensor
    data["visibs"] = vis_pred
    if conf_depth is not None:
        data["unc_metric"] = conf_depth

    save_path = os.path.join(output_dir, f"{filename}.npz")
    np.savez(save_path, **data)
    print(f"Results saved to {save_path}")
