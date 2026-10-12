"""
阶段二: Future-Visible 追踪
加载第一阶段结果，检测后续帧中新出现的未覆盖区域，
对这些 future-visible 点进行追踪，生成 future-visible tracking video。
"""
import os
import sys
import argparse
import numpy as np
import torch

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.SpaTrackV2.utils.visualizer import Visualizer
from tracking_video.utils import (load_models, sample_future_visible_queries,
                           sample_future_visible_queries_v2,
                           sample_future_visible_queries_inverse,
                           save_tracking_results, filter_tracks_by_quality,
                           resample_track_depth)
from tracking_video.das_visualizer import DaSVisualizer


# ============================================================
# 任务 2.1: 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Future-visible tracking")
    parser.add_argument("--stage1_dir", type=str, required=True,
                        help="Directory containing stage1_results.npz")
    parser.add_argument("--grid_size", type=int, default=50,
                        help="Grid size for future-visible sampling (independent of stage 1)")
    parser.add_argument("--vo_points", type=int, default=384)
    parser.add_argument("--track_mode", type=str, default="offline", choices=["offline", "online"])
    parser.add_argument("--vis_threshold", type=float, default=0.7,
                        help="Quality filter: visibility threshold")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Quality filter: confidence threshold")
    parser.add_argument("--coverage_radius", type=int, default=None,
                        help="Coverage radius in pixels (default: auto based on grid_size)")
    parser.add_argument("--batch_size", type=int, default=2500,
                        help="Max queries per inference batch")
    parser.add_argument("--use_hybrid_coverage", action="store_true", default=True,
                        help="Use hybrid coverage (geometric projection + stage1 tracks)")
    parser.add_argument("--no_hybrid_coverage", dest="use_hybrid_coverage", action="store_false")
    parser.add_argument("--conf_threshold_depth", type=float, default=0.0,
                        help="Depth confidence threshold for geometric projection (0.0 = 关掉)")
    parser.add_argument("--dilate_pixels", type=int, default=1,
                        help="Dilation pixels for geometric projection coverage")
    parser.add_argument("--depth_margin", type=float, default=1.0,
                        help="Z-Buffer depth consistency margin (1.0 = 要求 proj_d > actual_0 * 2.0 才算遮挡)")
    parser.add_argument("--stage1_filter_radius", type=float, default=5.0,
                        help="方案 F': 候选 future-visible 距离 stage1 高质量追踪点 < radius (像素) 则丢弃; 0 = 关闭过滤")
    parser.add_argument("--das_fps", type=int, default=8,
                        help="DaS tracking video FPS")
    parser.add_argument("--das_point_size", type=int, default=0,
                        help="DaS rectangle half-size (0=adaptive)")
    parser.add_argument("--skip_das", action="store_true",
                        help="Skip DaS-style tracking video generation")
    parser.add_argument("--tracker", type=str, default="v2", choices=["v1", "v2"],
                        help="Tracker to use: v1 (SpatialTracker) or v2 (SpatialTrackerV2)")
    return parser.parse_args()


def run_future_visible_tracking(stage1_dir, grid_size=50, vo_points=756,
                                track_mode="offline", vis_threshold=1.1,
                                coverage_radius=None,
                                batch_size=2500, conf_threshold=0.9,
                                tracker_model=None, stage1_data=None,
                                use_hybrid_coverage=True,
                                conf_threshold_depth=0.0,
                                dilate_pixels=1,
                                depth_margin=1.0,
                                stage1_filter_radius=5.0,
                                das_fps=8, das_point_size=0, skip_das=False,
                                tracker="v2",
                                shared_depth=None):
    """
    执行第二阶段追踪，可以被 run_all.py 直接调用。
    如果传入 tracker_model 则复用，否则自行加载。
    如果传入 stage1_data (dict) 则直接使用，否则从 npz 加载。

    hybrid 模式新增参数:
        shared_depth: (T, H, W) ndarray — 共享深度 (MoGe), DaS 上色用
    """
    output_dir = stage1_dir

    # ============================================================
    # 任务 2.1 (续): 加载第一阶段结果
    # ============================================================
    if stage1_data is None:
        print("[Stage 2] Loading stage 1 results...")
        stage1_path = os.path.join(stage1_dir, "stage1_results.npz")
        data = np.load(stage1_path, allow_pickle=True)
        track2d_pred_s1 = data["track2d_pred"]
        vis_pred_s1 = data["vis_pred"]
        conf_pred_s1 = data["conf_pred"] if "conf_pred" in data else None
        video_tensor = data["video_tensor"]
        depth_tensor = data["depth_tensor"] if "depth_tensor" in data else None
        intrs = data["intrs"]
        extrs = data["extrs"]
        unc_metric = data["unc_metric"] if "unc_metric" in data else None
        frame_H = int(data["frame_H"])
        frame_W = int(data["frame_W"])
        orig_H = int(data["orig_H"]) if "orig_H" in data else frame_H
        orig_W = int(data["orig_W"]) if "orig_W" in data else frame_W
        video_for_viz = data["video"]  # noqa: F841
        # 几何投影所需的额外数据
        point_map = data["point_map"] if "point_map" in data else None
        c2w_traj_s1 = data["c2w_traj"] if "c2w_traj" in data else None
        conf_depth_s1 = data["conf_depth"] if "conf_depth" in data else None
    else:
        print("[Stage 2] Using stage 1 results from memory...")
        track2d_pred_s1 = stage1_data["track2d_pred"]
        vis_pred_s1 = stage1_data["vis_pred"]
        conf_pred_s1 = stage1_data.get("conf_pred")
        video_tensor = stage1_data["video_tensor"]
        depth_tensor = stage1_data.get("depth_tensor")
        intrs = stage1_data["intrs"]
        extrs = stage1_data["extrs"]
        unc_metric = stage1_data.get("unc_metric")
        frame_H = stage1_data["frame_H"]
        frame_W = stage1_data["frame_W"]
        orig_H = stage1_data.get("orig_H", frame_H)
        orig_W = stage1_data.get("orig_W", frame_W)
        video_for_viz = stage1_data["video"]
        # 几何投影所需的额外数据
        point_map = stage1_data.get("point_map")
        c2w_traj_s1 = stage1_data.get("c2w_traj")
        conf_depth_s1 = stage1_data.get("conf_depth")

    T = video_tensor.shape[0]
    print(f"[Stage 2] Video: {T} frames, {frame_H}x{frame_W}")

    # ============================================================
    # 任务 2.2: 检测 Future-Visible 网格点 (反向投影: frame t → frame 0)
    # ============================================================
    # 确保数据是 numpy
    if isinstance(track2d_pred_s1, torch.Tensor):
        track2d_pred_s1 = track2d_pred_s1.cpu().numpy()
    if isinstance(vis_pred_s1, torch.Tensor):
        vis_pred_s1 = vis_pred_s1.cpu().numpy()
    if conf_pred_s1 is not None and isinstance(conf_pred_s1, torch.Tensor):
        conf_pred_s1 = conf_pred_s1.cpu().numpy()

    print(f"[Stage 2] Quality filter: vis>{vis_threshold}, conf>{conf_threshold}")

    if point_map is not None and c2w_traj_s1 is not None:
        print("[Stage 2] Using INVERSE projection (frame t → frame 0) to detect future-visible points...")
        future_queries = sample_future_visible_queries_inverse(
            point_map=point_map,
            c2w_traj=c2w_traj_s1,
            intrs=intrs,
            grid_size=grid_size,
            frame_H=frame_H,
            frame_W=frame_W,
            conf_depth=conf_depth_s1,
            conf_threshold_t=conf_threshold_depth,
            depth_margin=depth_margin,
            track2d_s1=track2d_pred_s1,
            vis_pred_s1=vis_pred_s1,
            stage1_filter_radius=stage1_filter_radius,
        )
    else:
        print("[Stage 2] WARNING: point_map/c2w_traj not available. Falling back to legacy coverage.")
        if coverage_radius is None:
            coverage_radius = max(2, min(frame_H, frame_W) // (grid_size * 2))
        future_queries = sample_future_visible_queries(
            track2d_pred_s1, vis_pred_s1, grid_size, frame_H, frame_W,
            coverage_vis_threshold=0.5, coverage_radius=coverage_radius
        )
    print(f"[Stage 2] Future-visible query points: {len(future_queries)}")

    if len(future_queries) == 0:
        print("[Stage 2] No future-visible points found. All pixels are covered by stage 1.")
        return None

    # 统计查询点的来源帧分布
    unique_frames, counts = np.unique(future_queries[:, 0].astype(int), return_counts=True)
    print(f"[Stage 2] Queries spread across {len(unique_frames)} frames:")
    for f, c in zip(unique_frames[:10], counts[:10]):
        print(f"  Frame {f}: {c} points")
    if len(unique_frames) > 10:
        print(f"  ... and {len(unique_frames) - 10} more frames")

    # ============================================================
    # 任务 2.4: 加载模型并运行追踪
    # ============================================================
    if tracker_model is None:
        _, tracker_model = load_models(track_mode, vo_points, tracker=tracker)
    elif tracker == "v2":
        tracker_model.spatrack.track_num = vo_points

    # 确保 video_tensor 是 torch.Tensor
    if isinstance(video_tensor, np.ndarray):
        video_tensor = torch.from_numpy(video_tensor).float()

    # V2 每次 forward 最多处理 track_num 个查询点, 超过会 shape 不匹配
    if tracker == "v2":
        batch_size = min(batch_size, vo_points)

    # ============================================================
    # 任务 2.3: 分批推理
    # ============================================================
    def _run_tracker_forward(tracker_model, video_tensor, depth_tensor, intrs, extrs,
                             queries, unc_metric, tracker):
        """统一的 tracker 前向推理调用"""
        if tracker == "v1":
            return tracker_model.forward(
                video_tensor, depth=depth_tensor,
                intrs=intrs, extrs=extrs,
                queries=queries,
                unc_metric=unc_metric,
            )
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                return tracker_model.forward(
                    video_tensor, depth=depth_tensor,
                    intrs=intrs, extrs=extrs,
                    queries=queries,
                    fps=1, full_point=False, iters_track=4,
                    query_no_BA=True, fixed_cam=False, stage=1,
                    unc_metric=unc_metric,
                    support_frame=len(video_tensor) - 1,
                    replace_ratio=0.2,
                    preserve_query_order=True,
                )

    if len(future_queries) <= batch_size:
        # 单次推理
        print(f"[Stage 2] Running future-visible tracking (single batch, tracker={tracker})...")
        (
            c2w_traj, intrs_out, point_map, conf_depth,
            track3d_pred_fv, track2d_pred_fv, vis_pred_fv, conf_pred_fv, video
        ) = _run_tracker_forward(tracker_model, video_tensor, depth_tensor, intrs, extrs,
                                  future_queries, unc_metric, tracker)
    else:
        # 分批推理
        num_batches = (len(future_queries) + batch_size - 1) // batch_size
        print(f"[Stage 2] Running future-visible tracking ({num_batches} batches, tracker={tracker})...")

        all_track3d = []
        all_track2d = []
        all_vis = []
        all_conf = []
        c2w_traj = None
        intrs_out = None
        point_map = None
        conf_depth = None
        video = None

        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, len(future_queries))
            batch_queries = future_queries[start:end]
            print(f"  Batch {b + 1}/{num_batches}: {len(batch_queries)} queries")

            (
                c2w_b, intrs_b, pm_b, cd_b,
                t3d_b, t2d_b, vis_b, conf_b, vid_b
            ) = _run_tracker_forward(tracker_model, video_tensor, depth_tensor, intrs, extrs,
                                      batch_queries, unc_metric, tracker)

            all_track3d.append(t3d_b.cpu() if isinstance(t3d_b, torch.Tensor) else torch.from_numpy(t3d_b))
            all_track2d.append(t2d_b.cpu() if isinstance(t2d_b, torch.Tensor) else torch.from_numpy(t2d_b))
            all_vis.append(vis_b.cpu() if isinstance(vis_b, torch.Tensor) else torch.from_numpy(vis_b))
            all_conf.append(conf_b.cpu() if isinstance(conf_b, torch.Tensor) else torch.from_numpy(conf_b))
            # 只取第一个 batch 的几何结果 (每次都一样)
            if c2w_traj is None:
                c2w_traj = c2w_b
                intrs_out = intrs_b
                point_map = pm_b
                conf_depth = cd_b
                video = vid_b

            torch.cuda.empty_cache()

        # 在 N 维度拼接
        track3d_pred_fv = torch.cat(all_track3d, dim=1)
        track2d_pred_fv = torch.cat(all_track2d, dim=1)
        vis_pred_fv = torch.cat(all_vis, dim=1)
        conf_pred_fv = torch.cat(all_conf, dim=1)

    print(f"[Stage 2] Tracking done. track2d: {track2d_pred_fv.shape}, vis: {vis_pred_fv.shape}")

    # ============================================================
    # 任务 2.5: 质量过滤 + 生成 Future-Visible Tracking Video
    # ============================================================
    # 质量过滤: 只保留高质量追踪点
    track3d_fv_filtered, track2d_fv_filtered, vis_fv_filtered, conf_fv_filtered, n_kept, keep_mask = \
        filter_tracks_by_quality(track3d_pred_fv, track2d_pred_fv, vis_pred_fv, conf_pred_fv,
                                 vis_threshold=vis_threshold, conf_threshold=conf_threshold)

    # 对齐 future_queries 和过滤后的 tracks (用 keep_mask 过滤)
    future_queries_filtered = future_queries[keep_mask]

    # 用原始 video 进行可视化
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    if isinstance(track2d_fv_filtered, np.ndarray):
        track2d_fv_filtered = torch.from_numpy(track2d_fv_filtered)
    if isinstance(vis_fv_filtered, np.ndarray):
        vis_fv_filtered = torch.from_numpy(vis_fv_filtered)

    # 创建可视化用的副本 (不影响保存的原始数据)
    track2d_for_viz = track2d_fv_filtered.clone()
    vis_for_viz = (vis_fv_filtered > 0.5).float()

    # 核心修复: 根据每个点的 query frame, mask 掉该帧之前的 visibility 和 track 坐标
    # 防止 future visible 点在其出现之前的帧被绘制 (反向预测的干扰)
    query_frames = future_queries_filtered[:, 0].astype(int)
    for i in range(n_kept):
        qf = query_frames[i]
        if qf > 0:
            vis_for_viz[:qf, i, :] = 0.0
            # 将 query frame 之前的坐标填充为 query frame 的位置,
            # 避免 track line 从错误位置画出 (Visualizer 画线不检查 visibility)
            track2d_for_viz[:qf, i, :] = track2d_for_viz[qf, i, :]

    print(f"[Stage 2] Masked pre-query-frame visibility for {(query_frames > 0).sum()} points "
          f"(query frames range: {query_frames.min()}-{query_frames.max()})")

    viser = Visualizer(save_dir=output_dir, grayscale=True,
                       fps=10, pad_value=0, tracks_leave_trace=5)
    viser.visualize(video=video[None],
                    tracks=track2d_for_viz[None][..., :2],
                    visibility=vis_for_viz[None],
                    filename="future_visible_tracking")
    print(f"[Stage 2] Future-visible tracking video saved to {output_dir}/future_visible_tracking_pred_track.mp4")

    # DaS 风格 tracking video (RGB编码空间位置)
    if not skip_das:
        # 使用已 mask 的可视化副本 (与 Visualizer 一致)
        tracks_3d_np = track2d_for_viz.cpu().numpy() if isinstance(track2d_for_viz, torch.Tensor) else np.array(track2d_for_viz)
        vis_np = vis_for_viz.cpu().numpy() if isinstance(vis_for_viz, torch.Tensor) else np.array(vis_for_viz)

        # hybrid 模式: 用 MoGe 共享深度替换 track depth, 确保与 Stage 1 DaS 颜色一致
        if shared_depth is not None:
            shared_depth_np = shared_depth.cpu().numpy() if isinstance(shared_depth, torch.Tensor) else shared_depth
            tracks_3d_np = resample_track_depth(tracks_3d_np, shared_depth_np)

        # 将追踪坐标从模型分辨率缩放到原始分辨率
        tracks_3d_np_scaled = tracks_3d_np.copy()
        tracks_3d_np_scaled[..., 0] *= orig_W / frame_W
        tracks_3d_np_scaled[..., 1] *= orig_H / frame_H

        das_viser = DaSVisualizer(save_dir=output_dir, fps=das_fps,
                                  point_size=das_point_size,
                                  vis_threshold=0.5)
        das_viser.generate(
            tracks_3d=tracks_3d_np_scaled,   # (T, N, 3)
            visibility=vis_np,               # (T, N, 1)
            H=orig_H, W=orig_W,
            filename="future_visible_tracking_das"
        )
        print(f"[Stage 2] DaS tracking video saved to {output_dir}/future_visible_tracking_das.mp4")

    # ============================================================
    # 任务 2.6: 保存结果 (过滤后的高质量结果)
    # ============================================================
    save_tracking_results(
        output_dir, "result_future_visible",
        c2w_traj, intrs_out, point_map, conf_depth,
        track3d_fv_filtered, track2d_fv_filtered, vis_fv_filtered, video_tensor
    )

    # 保存元数据
    meta_path = os.path.join(output_dir, "future_visible_meta.npz")
    np.savez(meta_path,
             future_queries=future_queries,
             coverage_radius=coverage_radius,
             vis_threshold=vis_threshold,
             conf_threshold=conf_threshold,
             grid_size=grid_size)
    print(f"[Stage 2] Metadata saved to {meta_path}")

    torch.cuda.empty_cache()
    print("[Stage 2] Done.")


if __name__ == "__main__":
    args = parse_args()
    run_future_visible_tracking(
        stage1_dir=args.stage1_dir,
        grid_size=args.grid_size,
        vo_points=args.vo_points,
        track_mode=args.track_mode,
        vis_threshold=args.vis_threshold,
        coverage_radius=args.coverage_radius,
        batch_size=args.batch_size,
        conf_threshold=args.conf_threshold,
        use_hybrid_coverage=args.use_hybrid_coverage,
        conf_threshold_depth=args.conf_threshold_depth,
        dilate_pixels=args.dilate_pixels,
        depth_margin=args.depth_margin,
        stage1_filter_radius=args.stage1_filter_radius,
        das_fps=args.das_fps,
        das_point_size=args.das_point_size,
        skip_das=args.skip_das,
        tracker=args.tracker,
    )
