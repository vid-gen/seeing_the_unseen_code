"""
阶段一: First Frame Visible 追踪
在第一帧上密集采样网格点，运行 SpaTracker 追踪，
生成 visible tracking video 并保存中间结果供第二阶段使用。
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
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from tracking_video.utils import load_video_data, load_models, save_tracking_results, filter_tracks_by_quality, resample_track_depth
from tracking_video.das_visualizer import DaSVisualizer


# ============================================================
# 任务 1.1: 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Visible tracking from first frame")
    parser.add_argument("--data_type", type=str, default="RGBD", choices=["RGB", "RGBD"])
    parser.add_argument("--data_dir", type=str, default="assets/example1")
    parser.add_argument("--video_name", type=str, default="snowboard")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--vo_points", type=int, default=384)
    parser.add_argument("--track_mode", type=str, default="offline", choices=["offline", "online"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {data_dir}/my_results)")
    parser.add_argument("--vis_threshold", type=float, default=0.7,
                        help="Quality filter: visibility threshold")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Quality filter: confidence threshold")
    parser.add_argument("--das_fps", type=int, default=8,
                        help="DaS tracking video FPS")
    parser.add_argument("--das_point_size", type=int, default=0,
                        help="DaS rectangle half-size (0=adaptive)")
    parser.add_argument("--skip_das", action="store_true",
                        help="Skip DaS-style tracking video generation")
    parser.add_argument("--tracker", type=str, default="v2", choices=["v1", "v2"],
                        help="Tracker to use: v1 (SpatialTracker) or v2 (SpatialTrackerV2)")
    parser.add_argument("--v1_checkpoint", type=str, default=None)
    parser.add_argument("--v1_seq_length", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2500,
                        help="Max queries per inference batch (V2 only)")
    return parser.parse_args()


def run_visible_tracking(data_type, data_dir, video_name, fps, grid_size,
                         vo_points, track_mode, output_dir,
                         vis_threshold=1.1, conf_threshold=0.9,
                         das_fps=8, das_point_size=0, skip_das=False,
                         vggt_model=None, tracker_model=None,
                         tracker="v2",
                         use_shared_depth=False,
                         batch_size=2500):
    """
    执行第一阶段追踪，可以被 run_all.py 直接调用。
    如果传入 vggt_model 和 tracker_model 则复用，否则自行加载。

    hybrid 模式新增参数:
        use_shared_depth: bool — 如果 True, DaS 上色用 VGGT depth_tensor
                          替换模型内部预测的 track depth, 确保颜色一致
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, "my_results")
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 任务 1.2: 模型初始化
    # ============================================================
    if vggt_model is None or tracker_model is None:
        vggt_model, tracker_model = load_models(track_mode, vo_points, tracker=tracker)
    elif tracker == "v2":
        tracker_model.spatrack.track_num = vo_points

    # ============================================================
    # 任务 1.1 (续): 视频加载
    # ============================================================
    print("[Stage 1] Loading video data...")
    video_tensor, depth_tensor, intrs, extrs, unc_metric, orig_H, orig_W = load_video_data(
        data_type, data_dir, video_name, fps, vggt_model
    )
    frame_H, frame_W = video_tensor.shape[2:]
    T = video_tensor.shape[0]
    print(f"[Stage 1] Video loaded: {T} frames, {frame_H}x{frame_W}")

    # ============================================================
    # 任务 1.3: 构造 First Frame 查询点
    # ============================================================
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    # query_xyt: (N, 3), 每行 [t=0, x, y]
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    print(f"[Stage 1] Query points: {query_xyt.shape[0]} points on first frame (grid_size={grid_size})")

    # ============================================================
    # 任务 1.4: 运行第一轮追踪（V2 支持分批以避免 OOM；V1 由内部自己生成网格，不分批）
    # ============================================================
    def _run_tracker_forward(queries):
        """统一的 tracker 前向推理调用（与 Stage 2 对齐）"""
        if tracker == "v1":
            return tracker_model.forward(
                video_tensor, depth=depth_tensor,
                intrs=intrs, extrs=extrs,
                queries=queries,
                grid_size=grid_size,
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

    if tracker == "v1":
        # V1: queries=None 让 V1 内部用 grid_size 在 interp_shape 上生成网格（与 DaS 一致），不分批
        print(f"[Stage 1] Running visible tracking (tracker=v1)...")
        (
            c2w_traj, intrs_out, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = _run_tracker_forward(None)
    else:
        # V2: 每次 forward 最多处理 vo_points 个查询点，超过会 shape 不匹配
        batch_size_eff = min(batch_size, vo_points)
        N = len(query_xyt)

        if N <= batch_size_eff:
            # 单次推理
            print(f"[Stage 1] Running visible tracking (single batch, {N} queries, tracker=v2)...")
            (
                c2w_traj, intrs_out, point_map, conf_depth,
                track3d_pred, track2d_pred, vis_pred, conf_pred, video
            ) = _run_tracker_forward(query_xyt)
        else:
            # 分批推理
            num_batches = (N + batch_size_eff - 1) // batch_size_eff
            print(f"[Stage 1] Running visible tracking ({num_batches} batches, {N} queries total, tracker=v2)...")

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
                start = b * batch_size_eff
                end = min(start + batch_size_eff, N)
                batch_queries = query_xyt[start:end]
                print(f"  Batch {b + 1}/{num_batches}: {len(batch_queries)} queries")

                (
                    c2w_b, intrs_b, pm_b, cd_b,
                    t3d_b, t2d_b, vis_b, conf_b, vid_b
                ) = _run_tracker_forward(batch_queries)

                all_track3d.append(t3d_b.cpu() if isinstance(t3d_b, torch.Tensor) else torch.from_numpy(t3d_b))
                all_track2d.append(t2d_b.cpu() if isinstance(t2d_b, torch.Tensor) else torch.from_numpy(t2d_b))
                all_vis.append(vis_b.cpu() if isinstance(vis_b, torch.Tensor) else torch.from_numpy(vis_b))
                all_conf.append(conf_b.cpu() if isinstance(conf_b, torch.Tensor) else torch.from_numpy(conf_b))
                # 几何结果（每批都一样）只取第一批
                if c2w_traj is None:
                    c2w_traj = c2w_b
                    intrs_out = intrs_b
                    point_map = pm_b
                    conf_depth = cd_b
                    video = vid_b

                torch.cuda.empty_cache()

            # 在 N 维度拼接
            track3d_pred = torch.cat(all_track3d, dim=1)
            track2d_pred = torch.cat(all_track2d, dim=1)
            vis_pred = torch.cat(all_vis, dim=1)
            conf_pred = torch.cat(all_conf, dim=1)

    # 修正 Frame 0 的坐标: V2 模型内部索引重排导致查询帧坐标错位，强制恢复为原始查询位置
    # V1 不需要此修正 (V1 predictor 内部已自行修正)
    if tracker == "v2":
        track2d_pred[0, :, :2] = torch.from_numpy(query_xyt[:, 1:]).to(track2d_pred.device)

    # V1 模式下，query_xyt 需要从 track2d_pred 的 frame 0 重建（因为网格由 V1 自己生成）
    if tracker == "v1":
        n_points = track2d_pred.shape[1]
        query_xyt = np.zeros((n_points, 3), dtype=np.float32)
        query_xyt[:, 1:] = track2d_pred[0, :, :2].cpu().numpy()

    print(f"[Stage 1] Tracking done. track2d: {track2d_pred.shape}, vis: {vis_pred.shape}")

    # ============================================================
    # 任务 1.5: 生成 Visible Tracking Video (直接用原始 vis_pred，阈值 0.8)
    # ============================================================
    viser = Visualizer(save_dir=output_dir, grayscale=True,
                       fps=10, pad_value=0, tracks_leave_trace=5)
    viser.visualize(video=video[None],
                    tracks=track2d_pred[None][..., :2],
                    visibility=vis_pred[None],
                    filename="visible_tracking")
    print(f"[Stage 1] Visible tracking video saved to {output_dir}/visible_tracking_pred_track.mp4")

    # DaS 风格 tracking video (RGB编码空间位置)
    if not skip_das:
        tracks_for_das = track2d_pred.cpu().numpy()
        if use_shared_depth:
            tracks_for_das = resample_track_depth(tracks_for_das, depth_tensor)
        # 将追踪坐标从模型分辨率缩放到原始分辨率
        tracks_for_das_scaled = tracks_for_das.copy()
        tracks_for_das_scaled[..., 0] *= orig_W / frame_W
        tracks_for_das_scaled[..., 1] *= orig_H / frame_H
        das_viser = DaSVisualizer(save_dir=output_dir, fps=das_fps,
                                  point_size=das_point_size,
                                  vis_threshold=0.9)
        das_viser.generate(
            tracks_3d=tracks_for_das_scaled,          # (T, N, 3)
            visibility=vis_pred.cpu().numpy(),        # (T, N, 1)
            H=orig_H, W=orig_W,
            filename="visible_tracking_das"
        )
        print(f"[Stage 1] DaS tracking video saved to {output_dir}/visible_tracking_das.mp4")

    # ============================================================
    # 任务 1.6: 保存结果
    # ============================================================
    # 保存 tapip3d 格式 (原始追踪结果，不做质量过滤)
    save_tracking_results(
        output_dir, "result_visible",
        c2w_traj, intrs_out, point_map, conf_depth,
        track3d_pred, track2d_pred, vis_pred, video_tensor
    )

    # 保存完整中间结果供第二阶段使用 (未过滤的原始数据，Stage 2 需要完整数据做覆盖判断)
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    stage1_data = {
        "track2d_pred": to_np(track2d_pred),  # 未过滤，Stage 2 覆盖判断用
        "track3d_pred": to_np(track3d_pred),
        "vis_pred": to_np(vis_pred),           # 未过滤，Stage 2 覆盖判断用
        "conf_pred": to_np(conf_pred),
        "video": to_np(video),
        "video_tensor": to_np(video_tensor),
        "depth_tensor": to_np(depth_tensor) if depth_tensor is not None else None,
        "intrs": to_np(intrs_out),
        "extrs": to_np(extrs),
        "c2w_traj": to_np(c2w_traj),
        "point_map": to_np(point_map),
        "conf_depth": to_np(conf_depth),
        "unc_metric": to_np(unc_metric) if unc_metric is not None else None,
        "query_xyt": query_xyt,
        "frame_H": frame_H,
        "frame_W": frame_W,
        "orig_H": orig_H,
        "orig_W": orig_W,
    }
    stage1_path = os.path.join(output_dir, "stage1_results.npz")
    np.savez(stage1_path, **{k: v for k, v in stage1_data.items() if v is not None})
    print(f"[Stage 1] Stage 1 results saved to {stage1_path}")

    torch.cuda.empty_cache()
    return stage1_data


if __name__ == "__main__":
    args = parse_args()
    run_visible_tracking(
        data_type=args.data_type,
        data_dir=args.data_dir,
        video_name=args.video_name,
        fps=args.fps,
        grid_size=args.grid_size,
        vo_points=args.vo_points,
        track_mode=args.track_mode,
        output_dir=args.output_dir,
        vis_threshold=args.vis_threshold,
        conf_threshold=args.conf_threshold,
        das_fps=args.das_fps,
        das_point_size=args.das_point_size,
        skip_das=args.skip_das,
        tracker=args.tracker,
        batch_size=args.batch_size,
    )
