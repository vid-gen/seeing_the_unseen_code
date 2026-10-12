"""
批量处理启动器 — 纯 V2 版本 (miradata1 专用)

与 batch_run.py 的区别:
  - 透传到 batch_worker_v2.py (纯 V2 模式, 而非 hybrid)
  - 默认参数对齐 miradata1 任务 (用户指定):
      --fps=1 --das_fps=24 --grid_size=70 --grid_size_fv=90
      --depth_margin=5.0 --das_point_size=2 --tracker=v2
      --track_mode=offline --vo_points=384 --conf_threshold_depth=0.0

扫描视频目录, 按GPU数量分割任务, 启动N个worker子进程,
等待完成, 合并日志, 生成汇总报告。
"""
import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Batch launcher V2: dispatch videos across GPUs (pure V2)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs (default: auto-detect)")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Number of worker processes per GPU (default: 1). "
                             "Total workers = num_gpus * workers_per_gpu. "
                             "增加此值前请确保单 worker 显存 * workers_per_gpu < 单卡显存。")
    parser.add_argument("--cpu_threads_per_worker", type=int, default=None,
                        help="CPU threads per worker (OMP/MKL/etc). "
                             "Default: auto = max(1, available_cpus // total_workers). "
                             "避免多 worker 互相抢 CPU 线程。")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Video source directory")
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Output root directory")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--das_fps", type=int, default=24)
    parser.add_argument("--grid_size", type=int, default=70)
    parser.add_argument("--grid_size_fv", type=int, default=90)
    parser.add_argument("--depth_margin", type=float, default=5.0)
    parser.add_argument("--das_point_size", type=int, default=2)
    parser.add_argument("--tracker", type=str, default="v2", choices=["v2"])
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--vo_points", type=int, default=384)
    parser.add_argument("--vis_threshold", type=float, default=0.9)
    parser.add_argument("--conf_threshold", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=2500)
    parser.add_argument("--conf_threshold_depth", type=float, default=0.0)
    parser.add_argument("--dilate_pixels", type=int, default=1)
    parser.add_argument("--stage1_filter_radius", type=float, default=5.0)
    return parser.parse_args()


def detect_gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


def scan_videos(data_dir):
    """扫描目录下所有 .mp4 文件, 返回排序后的视频名列表 (不含扩展名)"""
    videos = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".mp4"):
            videos.append(fname[:-4])
    videos.sort()
    return videos


def split_list(lst, n):
    """将列表均分为 n 组 (连续分割)"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def generate_summary(log_dir, total_workers, start_time):
    """合并所有 worker 的日志并生成汇总报告"""
    all_entries = []
    for wid in range(total_workers):
        log_path = os.path.join(log_dir, f"batch_log_w{wid}.jsonl")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_entries.append(json.loads(line))

    merged_path = os.path.join(log_dir, "batch_log_all.jsonl")
    with open(merged_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    success = [e for e in all_entries if e["status"] == "success"]
    failed = [e for e in all_entries if e["status"] == "failed"]
    skipped = [e for e in all_entries if e["status"] == "skipped"]

    end_time = datetime.now()
    elapsed_total = end_time - start_time

    if success:
        avg_time = sum(e["time_sec"] for e in success) / len(success)
    else:
        avg_time = 0

    lines = [
        "=" * 60,
        "批量处理汇总报告 (miradata1 纯 V2)",
        "=" * 60,
        f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时:   {elapsed_total}",
        f"Worker数: {total_workers}",
        "",
        f"总视频数: {len(all_entries)}",
        f"成功:     {len(success)}",
        f"失败:     {len(failed)}",
        f"跳过:     {len(skipped)}",
        f"平均耗时: {avg_time:.1f}s/video",
        "",
    ]

    if failed:
        lines.append("失败视频列表:")
        lines.append("-" * 40)
        for e in failed:
            lines.append(f"  {e['video']} - {e.get('error', 'unknown')}")
        lines.append("")

    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    summary_path = os.path.join(log_dir, "batch_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    return summary_text


def main():
    args = parse_args()
    start_time = datetime.now()

    num_gpus = args.num_gpus if args.num_gpus else detect_gpu_count()
    workers_per_gpu = args.workers_per_gpu
    total_workers = num_gpus * workers_per_gpu

    # 自动均分 CPU 线程, 避免多 worker 抢 CPU
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    if args.cpu_threads_per_worker is not None:
        threads_per_worker = args.cpu_threads_per_worker
    else:
        threads_per_worker = max(1, available_cpus // total_workers)
    print(f"[Launcher V2] Using {num_gpus} GPUs x {workers_per_gpu} workers/GPU = {total_workers} total workers")
    print(f"[Launcher V2] Available CPUs: {available_cpus}, threads/worker: {threads_per_worker}")

    log_dir = os.path.join(args.output_base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.output_base_dir, exist_ok=True)

    print(f"[Launcher V2] Scanning videos in {args.data_dir} ...")
    videos = scan_videos(args.data_dir)
    print(f"[Launcher V2] Found {len(videos)} videos")

    if not videos:
        print("[Launcher V2] No videos found. Exiting.")
        return

    splits = split_list(videos, total_workers)
    for wid, split in enumerate(splits):
        gpu_id = wid // workers_per_gpu
        list_path = os.path.join(log_dir, f"video_list_w{wid}.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(split) + "\n")
        print(f"[Launcher V2] Worker W{wid} (GPU {gpu_id}): {len(split)} videos -> {list_path}")

    passthrough_args = [
        f"--fps={args.fps}",
        f"--das_fps={args.das_fps}",
        f"--grid_size={args.grid_size}",
        f"--grid_size_fv={args.grid_size_fv}",
        f"--depth_margin={args.depth_margin}",
        f"--das_point_size={args.das_point_size}",
        f"--tracker={args.tracker}",
        f"--track_mode={args.track_mode}",
        f"--vo_points={args.vo_points}",
        f"--vis_threshold={args.vis_threshold}",
        f"--conf_threshold={args.conf_threshold}",
        f"--batch_size={args.batch_size}",
        f"--conf_threshold_depth={args.conf_threshold_depth}",
        f"--dilate_pixels={args.dilate_pixels}",
        f"--stage1_filter_radius={args.stage1_filter_radius}",
    ]

    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_worker_v2.py")
    processes = []
    for wid in range(total_workers):
        gpu_id = wid // workers_per_gpu
        list_path = os.path.join(log_dir, f"video_list_w{wid}.txt")
        log_path = os.path.join(log_dir, f"batch_log_w{wid}.jsonl")
        stdout_path = os.path.join(log_dir, f"stdout_w{wid}.log")

        cmd = [
            sys.executable, worker_script,
            f"--gpu_id={gpu_id}",
            f"--worker_id={wid}",
            f"--video_list={list_path}",
            f"--data_dir={args.data_dir}",
            f"--output_base_dir={args.output_base_dir}",
            f"--log_file={log_path}",
        ] + passthrough_args

        # GPU 隔离: 用 torch.cuda.set_device() (worker 内部处理),
        # 不再通过 CUDA_VISIBLE_DEVICES 隔离。
        worker_env = os.environ.copy()
        # 均分 CPU 线程, 避免 worker 间抢 OpenMP/MKL 池
        t = str(threads_per_worker)
        worker_env["OMP_NUM_THREADS"] = t
        worker_env["MKL_NUM_THREADS"] = t
        worker_env["OPENBLAS_NUM_THREADS"] = t
        worker_env["NUMEXPR_NUM_THREADS"] = t
        worker_env["VECLIB_MAXIMUM_THREADS"] = t

        stdout_file = open(stdout_path, "w")
        proc = subprocess.Popen(cmd, stdout=stdout_file, stderr=subprocess.STDOUT,
                                cwd=PROJECT_ROOT, env=worker_env)
        processes.append((wid, gpu_id, proc, stdout_file))
        print(f"[Launcher V2] Started W{wid} on GPU {gpu_id} (PID={proc.pid}, set_device={gpu_id})")

    print(f"\n[Launcher V2] All {total_workers} workers running. Waiting for completion...")
    for wid, gpu_id, proc, stdout_file in processes:
        retcode = proc.wait()
        stdout_file.close()
        status = "OK" if retcode == 0 else f"EXIT CODE {retcode}"
        print(f"[Launcher V2] W{wid} (GPU {gpu_id}) finished ({status})")

    print("\n[Launcher V2] Generating summary report...")
    summary = generate_summary(log_dir, total_workers, start_time)
    print(summary)


if __name__ == "__main__":
    main()
