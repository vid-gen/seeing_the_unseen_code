"""
单GPU批处理Worker — 纯 V2 版本 (miradata1 专用)

与 batch_worker.py 的区别:
  - 纯 V2 模式: Stage 1 和 Stage 2 都用 V2 tracker (不用 hybrid)
  - 输出文件重命名:
      visible_tracking_das.mp4        -> first_frame_tracking_video.mp4
      future_visible_tracking_das.mp4 -> future_visible_tracking_video.mp4
  - 只保留最终的两个 DaS mp4, 清理所有中间文件
  - 断点续传: 检查这两个最终文件是否存在且非空

通过 batch_run_v2.py 启动, 也可独立运行。
"""
import os
import sys
import argparse
import json
import time
import gc
import traceback

# GPU 隔离: 使用 torch.cuda.set_device(_GPU_ID)
# 必须在 import torch 之前解析 gpu_id, 并在 torch import 后立即 set_device
def _parse_gpu_id():
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--gpu_id="):
            return int(arg.split("=")[1])
        if arg == "--gpu_id" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 0

_GPU_ID = _parse_gpu_id()

import torch
torch.cuda.set_device(_GPU_ID)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tracking_video.utils import load_models
from tracking_video.run_visible_tracking import run_visible_tracking
from tracking_video.run_future_visible_tracking import run_future_visible_tracking


INTERMEDIATE_FILES = [
    "visible_tracking_pred_track.mp4",
    "future_visible_tracking_pred_track.mp4",
    "visible_tracking_das.mp4",
    "future_visible_tracking_das.mp4",
    "result_visible.npz",
    "result_future_visible.npz",
    "stage1_results.npz",
    "future_visible_meta.npz",
]

REQUIRED_OUTPUTS = [
    "first_frame_tracking_video.mp4",
    "future_visible_tracking_video.mp4",
]

RENAME_MAP = {
    "visible_tracking_das.mp4":        "first_frame_tracking_video.mp4",
    "future_visible_tracking_das.mp4": "future_visible_tracking_video.mp4",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch worker V2: pure V2 tracker, miradata1")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Physical GPU id (for logging only; real isolation via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Global worker id (for logging / distinguishing co-located workers)")
    parser.add_argument("--video_list", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_base_dir", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--das_fps", type=int, default=24)
    parser.add_argument("--grid_size", type=int, default=70)
    parser.add_argument("--grid_size_fv", type=int, default=90)
    parser.add_argument("--depth_margin", type=float, default=5.0)
    parser.add_argument("--das_point_size", type=int, default=2)
    parser.add_argument("--tracker", type=str, default="v2", choices=["v2"])
    parser.add_argument("--track_mode", type=str, default="offline", choices=["offline", "online"])
    parser.add_argument("--vo_points", type=int, default=384)
    parser.add_argument("--vis_threshold", type=float, default=0.9)
    parser.add_argument("--conf_threshold", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=2500)
    parser.add_argument("--conf_threshold_depth", type=float, default=0.0)
    parser.add_argument("--dilate_pixels", type=int, default=1)
    parser.add_argument("--stage1_filter_radius", type=float, default=5.0)
    return parser.parse_args()


def write_log(log_file, entry):
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def check_already_done(output_dir):
    """断点续传: 两个最终文件都存在且非空则视为已完成"""
    for fname in REQUIRED_OUTPUTS:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            return False
    return True


def rename_outputs(output_dir):
    """把 DaS 视频重命名为用户要求的最终文件名"""
    for src, dst in RENAME_MAP.items():
        sp = os.path.join(output_dir, src)
        dp = os.path.join(output_dir, dst)
        if os.path.exists(sp):
            if os.path.exists(dp):
                os.remove(dp)
            os.rename(sp, dp)


def cleanup_intermediate(output_dir):
    """删除所有中间文件, 只保留 first_frame_tracking_video.mp4 和 future_visible_tracking_video.mp4"""
    for fname in INTERMEDIATE_FILES:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)


def format_eta(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def process_single_video(video_name, args, vggt_model, v2_tracker):
    """纯 V2 模式: Stage 1 和 Stage 2 都用 v2_tracker"""
    output_dir = os.path.join(args.output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    stage1_data = run_visible_tracking(
        data_type="RGB",
        data_dir=args.data_dir,
        video_name=video_name,
        fps=args.fps,
        grid_size=args.grid_size,
        vo_points=args.vo_points,
        track_mode=args.track_mode,
        output_dir=output_dir,
        vis_threshold=args.vis_threshold,
        conf_threshold=args.conf_threshold,
        das_fps=args.das_fps,
        das_point_size=args.das_point_size,
        skip_das=False,
        vggt_model=vggt_model,
        tracker_model=v2_tracker,
        tracker="v2",
        batch_size=args.batch_size,
    )

    run_future_visible_tracking(
        stage1_dir=output_dir,
        grid_size=args.grid_size_fv,
        vo_points=args.vo_points,
        track_mode=args.track_mode,
        vis_threshold=args.vis_threshold,
        coverage_radius=None,
        batch_size=args.batch_size,
        conf_threshold=args.conf_threshold,
        tracker_model=v2_tracker,
        stage1_data=stage1_data,
        use_hybrid_coverage=True,
        conf_threshold_depth=args.conf_threshold_depth,
        dilate_pixels=args.dilate_pixels,
        depth_margin=args.depth_margin,
        stage1_filter_radius=args.stage1_filter_radius,
        das_fps=args.das_fps,
        das_point_size=args.das_point_size,
        skip_das=False,
        tracker="v2",
    )

    rename_outputs(output_dir)
    cleanup_intermediate(output_dir)


def main():
    args = parse_args()
    gpu_id = args.gpu_id  # 仅用于日志标识, CUDA 隔离由 CUDA_VISIBLE_DEVICES 负责
    wid = args.worker_id

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"[Worker W{wid} G{gpu_id}] CUDA_VISIBLE_DEVICES={visible}")
    print(f"[Worker W{wid} G{gpu_id}] torch.cuda.device_count()={torch.cuda.device_count()}")
    print(f"[Worker W{wid} G{gpu_id}] torch.cuda.current_device()={torch.cuda.current_device()}")
    if torch.cuda.device_count() > 0:
        print(f"[Worker W{wid} G{gpu_id}] device name: {torch.cuda.get_device_name(0)}")

    with open(args.video_list, "r") as f:
        video_names = [line.strip() for line in f if line.strip()]
    total = len(video_names)
    print(f"[Worker W{wid} G{gpu_id}] Assigned {total} videos")

    print(f"[Worker W{wid} G{gpu_id}] Loading models (pure V2)...")
    vggt_model, v2_tracker = load_models(args.track_mode, args.vo_points, tracker="v2")
    print(f"[Worker W{wid} G{gpu_id}] Models loaded successfully")

    success_count = 0
    failed_count = 0
    skipped_count = 0
    elapsed_times = []

    for i, video_name in enumerate(video_names):
        output_dir = os.path.join(args.output_base_dir, video_name)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if check_already_done(output_dir):
            skipped_count += 1
            write_log(args.log_file, {
                "video": video_name, "status": "skipped",
                "timestamp": timestamp, "gpu": gpu_id,
            })
            print(f"[Worker W{wid} G{gpu_id}] [{i+1}/{total}] {video_name} - SKIPPED (already done)")
            continue

        start_time = time.time()
        try:
            process_single_video(video_name, args, vggt_model, v2_tracker)
            elapsed = time.time() - start_time
            success_count += 1
            elapsed_times.append(elapsed)

            write_log(args.log_file, {
                "video": video_name, "status": "success",
                "time_sec": round(elapsed, 1),
                "timestamp": timestamp, "gpu": gpu_id,
            })

        except Exception as e:
            elapsed = time.time() - start_time
            failed_count += 1
            error_msg = str(e)
            tb = traceback.format_exc()

            write_log(args.log_file, {
                "video": video_name, "status": "failed",
                "time_sec": round(elapsed, 1),
                "error": error_msg,
                "traceback": tb,
                "timestamp": timestamp, "gpu": gpu_id,
            })
            print(f"[Worker W{wid} G{gpu_id}] [{i+1}/{total}] {video_name} - FAILED: {error_msg}")

            torch.cuda.empty_cache()
            gc.collect()
            continue

        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
        remaining = total - i - 1 - skipped_count
        eta = format_eta(avg_time * max(remaining, 0))
        print(f"[Worker W{wid} G{gpu_id}] [{i+1}/{total}] {video_name} - OK "
              f"({elapsed:.1f}s) | success={success_count} failed={failed_count} "
              f"skipped={skipped_count} | ETA: {eta}")

    print(f"\n[Worker W{wid} G{gpu_id}] ========== DONE ==========")
    print(f"[Worker W{wid} G{gpu_id}] Total: {total}")
    print(f"[Worker W{wid} G{gpu_id}] Success: {success_count}")
    print(f"[Worker W{wid} G{gpu_id}] Failed: {failed_count}")
    print(f"[Worker W{wid} G{gpu_id}] Skipped: {skipped_count}")
    if elapsed_times:
        print(f"[Worker W{wid} G{gpu_id}] Avg time per video: {sum(elapsed_times)/len(elapsed_times):.1f}s")


if __name__ == "__main__":
    main()
