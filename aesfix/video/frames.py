"""Fixed-rate frame extraction from a video file, via PyAV.

PyAV ships its own statically-linked FFmpeg libraries, so this needs no
system ffmpeg install (verified: this machine has none).
"""
import os
import tempfile
from dataclasses import dataclass
from typing import List

import av


@dataclass
class SampledFrame:
    index: int          # index among sampled frames (0-based)
    timestamp_s: float  # presentation time of the source frame, seconds
    path: str           # JPEG written to disk


def extract_frames(video_path: str, sample_rate_fps: float = 1.0, out_dir: str = None) -> List[SampledFrame]:
    """Decode video_path and write one JPEG per sampled frame to out_dir (a
    fresh temp directory if not given), sampling at a fixed rate.

    Sampling takes every Nth decoded frame, where N = round(native_fps /
    sample_rate_fps), clamped to >= 1 (never upsamples past the source
    frame rate).
    """
    if sample_rate_fps <= 0:
        raise ValueError(f"sample_rate_fps must be > 0, got {sample_rate_fps}")

    out_dir = out_dir or tempfile.mkdtemp(prefix="aesfix_frames_")
    os.makedirs(out_dir, exist_ok=True)

    container = av.open(video_path)
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate) if stream.average_rate else sample_rate_fps
    stride = max(1, round(native_fps / sample_rate_fps))

    sampled = []
    for decoded_index, frame in enumerate(container.decode(stream)):
        if decoded_index % stride != 0:
            continue
        if frame.pts is not None:
            timestamp_s = float(frame.pts * stream.time_base)
        else:
            timestamp_s = decoded_index / native_fps
        out_path = os.path.join(out_dir, f"frame_{len(sampled):04d}.jpg")
        frame.to_image().save(out_path, quality=95)
        sampled.append(SampledFrame(index=len(sampled), timestamp_s=round(timestamp_s, 3), path=out_path))

    container.close()
    return sampled
