import numpy as np
import cv2


def compute_das_colors(uv_frame0, depth_frame0):
    n = len(uv_frame0)
    colors = np.zeros((n, 3), dtype=np.uint8)

    x_vals = uv_frame0[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    if x_max > x_min:
        colors[:, 0] = (np.clip((x_vals - x_min) / (x_max - x_min), 0, 1) * 255).astype(np.uint8)
    else:
        colors[:, 0] = 128

    y_vals = uv_frame0[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    if y_max > y_min:
        colors[:, 1] = (np.clip((y_vals - y_min) / (y_max - y_min), 0, 1) * 255).astype(np.uint8)
    else:
        colors[:, 1] = 128

    z_vals = depth_frame0
    if np.all(z_vals == 0):
        colors[:, 2] = 128
    else:
        inv_z = 1.0 / (z_vals + 1e-10)
        p2, p98 = np.percentile(inv_z, 2), np.percentile(inv_z, 98)
        if p98 > p2:
            norm_z = np.clip((inv_z - p2) / (p98 - p2), 0, 1)
        else:
            norm_z = np.full_like(inv_z, 0.5)
        colors[:, 2] = (norm_z * 255).astype(np.uint8)

    return colors


def render_das_video(tracks_2d, tracks_depth, visibility, W, H, output_path,
                     fps=24, vis_threshold=0.9, point_size=2, lossless=False):
    n_steps, n_points, _ = tracks_2d.shape

    colors = compute_das_colors(tracks_2d[0], tracks_depth[0])

    import imageio
    from pathlib import Path
    output_path = Path(output_path)
    if lossless:
        output_path = output_path.with_suffix('.avi')
        writer = imageio.get_writer(str(output_path), fps=fps, codec='ffv1')
    else:
        writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264',
                                    pixelformat='yuv420p', quality=None,
                                    output_params=['-crf', '18'])

    for t in range(n_steps):
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        depth_t = tracks_depth[t]
        sort_idx = np.argsort(-depth_t)

        for n in sort_idx:
            if visibility[t, n] <= vis_threshold:
                continue
            x = int(round(tracks_2d[t, n, 0]))
            y = int(round(tracks_2d[t, n, 1]))
            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            color_bgr = (int(colors[n, 2]), int(colors[n, 1]), int(colors[n, 0]))
            x0 = max(0, x - point_size)
            y0 = max(0, y - point_size)
            x1 = min(W - 1, x + point_size)
            y1 = min(H - 1, y + point_size)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color_bgr, -1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()
    print(f"Saved DaS video: {output_path} ({n_steps} frames, {fps} fps)")
