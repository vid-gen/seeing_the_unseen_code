import numpy as np
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def render_rainbow_video(tracks_2d, visibility, first_frame, output_path,
                         fps=10, vis_threshold=0.9, trace_len=5, dot_radius=4,
                         tracks_depth=None):
    n_steps, n_points, _ = tracks_2d.shape
    H, W = first_frame.shape[:2]

    gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    gray_bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    y_coords = tracks_2d[0, :, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    if y_max <= y_min:
        y_max = y_min + 1.0
    norm = plt.Normalize(y_min, y_max)
    cmap = cm.get_cmap('gist_rainbow')

    colors = np.zeros((n_points, 3), dtype=np.uint8)
    for n in range(n_points):
        c = cmap(norm(y_coords[n]))
        colors[n] = (np.array(c[:3]) * 255).astype(np.uint8)

    import imageio
    writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264',
                                pixelformat='yuv420p', quality=None,
                                output_params=['-crf', '18'])

    for t in range(n_steps):
        frame = gray_bg.copy()

        if tracks_depth is not None:
            sort_idx = np.argsort(-tracks_depth[t])
        else:
            sort_idx = np.arange(n_points)

        for n in sort_idx:
            if visibility[t, n] <= vis_threshold:
                continue
            x, y = int(round(tracks_2d[t, n, 0])), int(round(tracks_2d[t, n, 1]))
            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            if trace_len > 0:
                for prev in range(max(0, t - trace_len), t):
                    if visibility[prev, n] <= vis_threshold:
                        continue
                    x0 = int(round(tracks_2d[prev, n, 0]))
                    y0 = int(round(tracks_2d[prev, n, 1]))
                    x1 = int(round(tracks_2d[min(prev + 1, t), n, 0]))
                    y1 = int(round(tracks_2d[min(prev + 1, t), n, 1]))
                    color_bgr = (int(colors[n, 2]), int(colors[n, 1]), int(colors[n, 0]))
                    cv2.line(frame, (x0, y0), (x1, y1), color_bgr, 1, cv2.LINE_AA)

            color_bgr = (int(colors[n, 2]), int(colors[n, 1]), int(colors[n, 0]))
            cv2.circle(frame, (x, y), dot_radius, color_bgr, -1, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()
    print(f"Saved rainbow video: {output_path} ({n_steps} frames, {fps} fps)")
