"""
一键运行入口脚本：
  阶段一: First Frame Visible 追踪
  阶段二: Future-Visible 追踪
共享模型加载，避免重复加载节省时间和显存。
"""
import os
import sys
import argparse

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tracking_video.utils import load_models
from tracking_video.run_visible_tracking import run_visible_tracking
from tracking_video.run_future_visible_tracking import run_future_visible_tracking


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run visible + future-visible tracking pipeline"
    )
    # 数据相关
    parser.add_argument("--data_type", type=str, default="RGBD", choices=["RGB", "RGBD"])
    parser.add_argument("--data_dir", type=str, default="assets/example1")
    parser.add_argument("--video_name", type=str, default="snowboard")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {data_dir}/my_results)")
    # 模型相关
    parser.add_argument("--tracker", type=str, default="v2", choices=["v1", "v2", "hybrid"],
                        help="Tracker: v1, v2, or hybrid (V1 visible + V2 future-visible)")
    parser.add_argument("--track_mode", type=str, default="offline", choices=["offline", "online"])
    parser.add_argument("--vo_points", type=int, default=384)
    # V1 特有参数
    parser.add_argument("--v1_checkpoint", type=str, default=None,
                        help="Path to SpaTracker V1 checkpoint (default: checkpoints/spaT_final.pth)")
    parser.add_argument("--v1_seq_length", type=int, default=12,
                        help="V1 sliding window length (default: 12)")
    # Stage 1 参数
    parser.add_argument("--grid_size", type=int, default=50)
    # Stage 2 独立网格密度 (不指定则与 stage 1 相同)
    parser.add_argument("--grid_size_fv", type=int, default=None,
                        help="Grid size for future-visible tracking (default: same as --grid_size)")
    # 质量过滤参数 (同时用于 Stage 1 和 Stage 2)
    parser.add_argument("--vis_threshold", type=float, default=0.9,
                        help="Quality filter: visibility threshold")
    parser.add_argument("--conf_threshold", type=float, default=0.9,
                        help="Quality filter: confidence threshold")
    parser.add_argument("--coverage_radius", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2500)
    # 混合覆盖方案参数 (Stage 2)
    parser.add_argument("--use_hybrid_coverage", action="store_true", default=True)
    parser.add_argument("--no_hybrid_coverage", dest="use_hybrid_coverage", action="store_false")
    parser.add_argument("--conf_threshold_depth", type=float, default=0.0,
                        help="Depth confidence threshold for geometric projection (0.0 = 关掉, 让所有 cam_z>0 的点都参与判定)")
    parser.add_argument("--dilate_pixels", type=int, default=1,
                        help="Dilation pixels for geometric projection coverage")
    parser.add_argument("--depth_margin", type=float, default=1.0,
                        help="Z-Buffer depth consistency margin (1.0 = 要求 proj_d > actual_0 * 2.0 才算遮挡, 减少 VGGT 深度抖动误判)")
    parser.add_argument("--stage1_filter_radius", type=float, default=5.0,
                        help="方案 F': 候选 future-visible 距离 stage1 高质量追踪点 < radius 像素则丢弃 (剔除动态物体的反投影误判). 0 = 关闭")
    # DaS 风格 tracking video 参数
    parser.add_argument("--das_fps", type=int, default=8,
                        help="DaS tracking video FPS")
    parser.add_argument("--das_point_size", type=int, default=0,
                        help="DaS rectangle half-size (0=adaptive)")
    parser.add_argument("--skip_das", action="store_true",
                        help="Skip DaS-style tracking video generation")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.data_dir, "my_results")
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 共享模型加载 (只加载一次)
    # ============================================================
    print("=" * 60)
    print(f"Loading models (tracker={args.tracker}, shared across both stages)...")
    print("=" * 60)

    # hybrid 模式: 加载 VGGT + V1 + V2 三个模型
    if args.tracker == "hybrid":
        vggt_model, v1_tracker, v2_tracker = load_models(
            args.track_mode, args.vo_points,
            tracker="hybrid",
            v1_checkpoint=args.v1_checkpoint,
            v1_seq_length=args.v1_seq_length,
        )
    else:
        vggt_model, tracker_model = load_models(
            args.track_mode, args.vo_points,
            tracker=args.tracker,
            v1_checkpoint=args.v1_checkpoint,
            v1_seq_length=args.v1_seq_length,
        )

    # ============================================================
    # 阶段一: Visible Tracking
    # ============================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Visible Tracking (first frame)")
    print("=" * 60)

    if args.tracker == "hybrid":
        # hybrid: Stage 1 用 V1, DaS 用 VGGT 共享深度上色
        stage1_data = run_visible_tracking(
            data_type=args.data_type,
            data_dir=args.data_dir,
            video_name=args.video_name,
            fps=args.fps,
            grid_size=args.grid_size,
            vo_points=args.vo_points,
            track_mode=args.track_mode,
            output_dir=output_dir,
            vis_threshold=args.vis_threshold,
            conf_threshold=args.conf_threshold,
            das_fps=args.das_fps,
            das_point_size=args.das_point_size,
            skip_das=args.skip_das,
            vggt_model=vggt_model,
            tracker_model=v1_tracker,
            tracker="v1",
            use_shared_depth=True,
            batch_size=args.batch_size,
        )
    else:
        stage1_data = run_visible_tracking(
            data_type=args.data_type,
            data_dir=args.data_dir,
            video_name=args.video_name,
            fps=args.fps,
            grid_size=args.grid_size,
            vo_points=args.vo_points,
            track_mode=args.track_mode,
            output_dir=output_dir,
            vis_threshold=args.vis_threshold,
            conf_threshold=args.conf_threshold,
            das_fps=args.das_fps,
            das_point_size=args.das_point_size,
            skip_das=args.skip_das,
            vggt_model=vggt_model,
            tracker_model=tracker_model,
            tracker=args.tracker,
            batch_size=args.batch_size,
        )

    # ============================================================
    # 阶段二: Future-Visible Tracking
    # ============================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Future-Visible Tracking")
    print("=" * 60)

    if args.tracker == "hybrid":
        # hybrid: Stage 2 用 V2, DaS 用 VGGT 共享深度上色
        run_future_visible_tracking(
            stage1_dir=output_dir,
            grid_size=args.grid_size_fv if args.grid_size_fv is not None else args.grid_size,
            vo_points=args.vo_points,
            track_mode=args.track_mode,
            vis_threshold=args.vis_threshold,
            coverage_radius=args.coverage_radius,
            batch_size=args.batch_size,
            conf_threshold=args.conf_threshold,
            tracker_model=v2_tracker,
            stage1_data=stage1_data,
            use_hybrid_coverage=args.use_hybrid_coverage,
            conf_threshold_depth=args.conf_threshold_depth,
            dilate_pixels=args.dilate_pixels,
            depth_margin=args.depth_margin,
            stage1_filter_radius=args.stage1_filter_radius,
            das_fps=args.das_fps,
            das_point_size=args.das_point_size,
            skip_das=args.skip_das,
            tracker="v2",
            shared_depth=stage1_data.get("depth_tensor"),
        )
    else:
        run_future_visible_tracking(
            stage1_dir=output_dir,
            grid_size=args.grid_size_fv if args.grid_size_fv is not None else args.grid_size,
            vo_points=args.vo_points,
            track_mode=args.track_mode,
            vis_threshold=args.vis_threshold,
            coverage_radius=args.coverage_radius,
            batch_size=args.batch_size,
            conf_threshold=args.conf_threshold,
            tracker_model=tracker_model,
            stage1_data=stage1_data,
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

    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "=" * 60)
    print("DONE! Output files:")
    print(f"  Visible tracking video:        {output_dir}/visible_tracking_pred_track.mp4")
    print(f"  Future-visible tracking video:  {output_dir}/future_visible_tracking_pred_track.mp4")
    if not args.skip_das:
        print(f"  DaS visible tracking video:     {output_dir}/visible_tracking_das.mp4")
        print(f"  DaS future-visible video:       {output_dir}/future_visible_tracking_das.mp4")
    print(f"  Visible results (tapip3d):      {output_dir}/result_visible.npz")
    print(f"  Future-visible results:         {output_dir}/result_future_visible.npz")
    print(f"  Stage 1 intermediate data:      {output_dir}/stage1_results.npz")
    print("=" * 60)


if __name__ == "__main__":
    main()
