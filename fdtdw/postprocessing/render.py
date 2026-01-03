import warp as wp
import numpy as np
import imageio
from fdtdw.kernels.structs32 import TEMStates

import warp as wp
import imageio


@wp.kernel
def render(
    field_data: wp.array3d(dtype=wp.float32),
    video_buffer: wp.array4d(dtype=wp.uint8),
    vmin: float,
    vmax: float,
    scale: int,
):
    t, u, v = wp.tid()

    x = u // scale

    h = field_data.shape[2]

    y = h - 1 - (v // scale)

    val = field_data[t, x, y]

    normalized = (val - vmin) / (vmax - vmin)
    t_val = wp.clamp(normalized, 0.0, 1.0)

    r = 0.0
    g = 0.0
    b = 0.0

    if t_val < 0.5:
        factor = t_val * 2.0
        r = 1.0
        g = factor
        b = factor
    else:
        factor = (t_val - 0.5) * 2.0
        r = 1.0 - factor
        g = 1.0 - factor
        b = 1.0

    video_buffer[t, v, u, 0] = wp.uint8(r * 255.0)
    video_buffer[t, v, u, 1] = wp.uint8(g * 255.0)
    video_buffer[t, v, u, 2] = wp.uint8(b * 255.0)


@wp.kernel
def slice(
    field_4d: wp.array4d(dtype=wp.float32),
    field_3d_seq: wp.array3d(dtype=wp.float32),
    slice_idx: int,
    axis: int,
):
    t, u, v = wp.tid()

    if axis == 3:
        field_3d_seq[t, u, v] = field_4d[t, u, v, slice_idx]

    elif axis == 2:
        field_3d_seq[t, u, v] = field_4d[t, u, slice_idx, v]

    elif axis == 1:
        field_3d_seq[t, u, v] = field_4d[t, slice_idx, u, v]


def render_array(
    field: wp.array3d(dtype=wp.float32),
    filename: str,
    fps: int = 30,
    limit: float = 0.1,
    scale: int = 4,
):
    T, W, H = field.shape

    Video_W = W * scale
    Video_H = H * scale

    video_buffer_device = wp.empty(
        (
            T,
            Video_H,
            Video_W,
            3,
        ),
        dtype=wp.uint8,
        device=field.device,
    )

    vmin, vmax = -limit, limit

    wp.launch(
        render,
        dim=(T, Video_W, Video_H),
        inputs=[field, video_buffer_device, vmin, vmax, scale],
        device=field.device,
    )

    video_data = video_buffer_device.numpy()

    imageio.mimsave(
        filename,
        list(video_data),
        fps=fps,
        codec="libx264",
        pixelformat="yuv444p",
        ffmpeg_params=["-crf", "0"],
        macro_block_size=1,
    )
