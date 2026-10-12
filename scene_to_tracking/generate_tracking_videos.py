import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_data
from mesh_sampler import sample_visible_tracking_points, sample_future_visible_tracking_points
from motion_tracker import compute_all_frames
from visibility import compute_visibility, apply_inter_object_occlusion
from rainbow_visualizer import render_rainbow_video
from das_visualizer import render_das_video
from time_resampler import build_time_mapping, resample_tracks, remap_query_frames


def apply_future_visible_masking(tracks_2d, tracks_depth, visibility, query_frames):
    _, n_points = visibility.shape
    tracks_2d = tracks_2d.copy()
    tracks_depth = tracks_depth.copy()
    visibility = visibility.copy()
    for i in range(n_points):
        qf = int(query_frames[i])
        if qf > 0:
            visibility[:qf, i] = 0.0
            tracks_2d[:qf, i] = tracks_2d[qf, i]
            tracks_depth[:qf, i] = tracks_depth[qf, i]
    return tracks_2d, tracks_depth, visibility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--grid_size', type=int, default=70)
    parser.add_argument('--dense_spacing', type=int, default=4)
    parser.add_argument('--video_fps', type=int, default=24)
    parser.add_argument('--video_frames', type=int, default=121)
    parser.add_argument('--sim_dt', type=float, default=0.03)
    parser.add_argument('--sim_duration', type=float, default=None,
                        help='Only use first N seconds of simulation (default: use all)')
    parser.add_argument('--das_point_size', type=int, default=2)
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix appended to output filenames before .mp4')
    parser.add_argument('--phase', type=int, default=0, help='0=all, 1=visible only, 2=future visible only')
    args = parser.parse_args()

    output_dir = Path(args.output_dir or args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Loading data ===")
    data = load_all_data(args.data_dir)
    print(f"  Rollout: {data['n_steps']} steps, {data['rollout_positions'].shape[1]} particles")
    print(f"  Meshes: {[len(m.vertices) for m in data['meshes']]} vertices")
    print(f"  Image: {data['first_frame'].shape}")
    W = data['camera']['image_width']
    H = data['camera']['image_height']

    print("\n=== Building time mapping ===")
    time_mapping = build_time_mapping(
        args.video_frames, args.sim_dt, data['n_steps'], args.sim_duration)
    sim_dur = args.sim_duration if args.sim_duration else (data['n_steps'] - 1) * args.sim_dt
    print(f"  Sim: {sim_dur:.2f}s ({data['n_steps']} steps @ {1/args.sim_dt:.1f} FPS) "
          f"-> {args.video_frames} video frames @ {args.video_fps} FPS")

    # ========== Phase 1: Visible Tracking ==========
    if args.phase in (0, 1):
        print("\n=== Phase 1: Visible Tracking ===")

        print("\n--- Sampling tracking points ---")
        sample_result = sample_visible_tracking_points(data, grid_size=args.grid_size)

        print("\n--- Computing per-frame positions (all sim steps) ---")
        tracks_result = compute_all_frames(sample_result, data)
        print(f"  tracks_2d shape: {tracks_result['tracks_2d'].shape}")

        print("\n--- Computing visibility (all sim steps) ---")
        visibility = compute_visibility(tracks_result, sample_result, data)

        print("\n--- Computing inter-object occlusion ---")
        visibility = apply_inter_object_occlusion(
            visibility, tracks_result['tracks_2d'], tracks_result['tracks_depth'],
            sample_result, data)

        print("\n--- Resampling to video frames ---")
        tracks_result, visibility = resample_tracks(
            tracks_result, visibility, time_mapping, sample_result, data)
        print(f"  Resampled tracks_2d shape: {tracks_result['tracks_2d'].shape}")
        vis_rate = (visibility > 0.9).mean()
        print(f"  Average visibility rate: {vis_rate:.1%}")

        print("\n--- Rendering rainbow video ---")
        render_rainbow_video(
            tracks_result['tracks_2d'], visibility, data['first_frame'],
            output_dir / f'visible_tracking_pred_track{args.suffix}.mp4', fps=args.video_fps,
            tracks_depth=tracks_result['tracks_depth'],
        )

        print("\n--- Rendering DaS video ---")
        render_das_video(
            tracks_result['tracks_2d'], tracks_result['tracks_depth'], visibility,
            W, H, output_dir / f'visible_tracking_das{args.suffix}.mp4',
            fps=args.video_fps, point_size=args.das_point_size,
        )

    # ========== Phase 2: Future Visible Tracking ==========
    if args.phase in (0, 2):
        print("\n=== Phase 2: Future Visible Tracking ===")

        print("\n--- Sampling future visible points ---")
        fv_result = sample_future_visible_tracking_points(data, dense_spacing=args.dense_spacing)

        if fv_result is not None and fv_result['n_points'] > 0:
            print("\n--- Computing per-frame positions (all sim steps) ---")
            fv_tracks = compute_all_frames(fv_result, data)
            print(f"  tracks_2d shape: {fv_tracks['tracks_2d'].shape}")

            print("\n--- Computing visibility (all sim steps) ---")
            fv_vis = compute_visibility(fv_tracks, fv_result, data)

            print("\n--- Computing inter-object occlusion ---")
            fv_vis = apply_inter_object_occlusion(
                fv_vis, fv_tracks['tracks_2d'], fv_tracks['tracks_depth'],
                fv_result, data)

            print("\n--- Resampling to video frames ---")
            fv_tracks, fv_vis = resample_tracks(
                fv_tracks, fv_vis, time_mapping, fv_result, data)
            print(f"  Resampled tracks_2d shape: {fv_tracks['tracks_2d'].shape}")

            print("\n--- Remapping query frames ---")
            query_frames_video = remap_query_frames(
                fv_result['query_frames'], args.sim_dt, args.video_frames,
                args.sim_duration, data['n_steps'])
            print(f"  Query frame range: sim [{fv_result['query_frames'].min()}, "
                  f"{fv_result['query_frames'].max()}] -> video [{query_frames_video.min()}, "
                  f"{query_frames_video.max()}]")

            print("\n--- Applying future visible masking ---")
            fv_tracks_2d, fv_tracks_depth, fv_vis = apply_future_visible_masking(
                fv_tracks['tracks_2d'], fv_tracks['tracks_depth'],
                fv_vis, query_frames_video,
            )
            vis_rate = (fv_vis > 0.5).mean()
            print(f"  Visibility rate (>0.5): {vis_rate:.1%}")

            print("\n--- Rendering rainbow video ---")
            render_rainbow_video(
                fv_tracks_2d, fv_vis, data['first_frame'],
                output_dir / f'future_visible_tracking_pred_track{args.suffix}.mp4',
                fps=args.video_fps, vis_threshold=0.9,
                tracks_depth=fv_tracks_depth,
            )

            print("\n--- Rendering DaS video ---")
            render_das_video(
                fv_tracks_2d, fv_tracks_depth, fv_vis,
                W, H, output_dir / f'future_visible_tracking_das{args.suffix}.mp4',
                fps=args.video_fps, point_size=args.das_point_size,
                vis_threshold=0.5,
            )
        else:
            print("  No future visible points found, skipping.")

    print("\n=== Done ===")
    print(f"Output files in: {output_dir}")
    print(f"Video spec: {args.video_frames} frames @ {args.video_fps} FPS, {W}x{H}")


if __name__ == '__main__':
    main()
