"""Score a video's aesthetic quality: sample frames at a fixed rate, run the
Phase-0 CLIP + fuzzy scorer (baselines/legacy_fuzzy, unmodified) on each, and
aggregate per-dimension mean/variance across frames.

extract_image_features raises ImageModelError on failure (Phase 0 bug 1 fix)
rather than returning fabricated defaults; this module does not catch it, so
a broken scorer fails the whole video instead of silently skipping frames.
"""
import argparse
import json
import statistics as stats

from baselines.legacy_fuzzy.our_system.fuzzy_system import build_fuzzy_system, evaluate_aesthetics
from baselines.legacy_fuzzy.our_system.image_model import extract_image_features

from .frames import extract_frames

DIMENSIONS = ["composition", "lighting", "color", "subject_clarity", "technical"]


def score_video(video_path: str, sample_rate_fps: float = 1.0) -> dict:
    frames = extract_frames(video_path, sample_rate_fps=sample_rate_fps)
    fuzzy_sim = build_fuzzy_system()

    per_frame = []
    for f in frames:
        features = extract_image_features(f.path)
        label, score = evaluate_aesthetics(
            fuzzy_sim,
            features["composition"], features["lighting"], features["color"],
            features["subject_clarity"], features["technical"],
        )
        per_frame.append({
            "frame_index": f.index,
            "timestamp_s": f.timestamp_s,
            **{d: features[d] for d in DIMENSIONS},
            "fuzzy_label": label,
            "fuzzy_score": score,
        })

    aggregated = {}
    for key in DIMENSIONS + ["fuzzy_score"]:
        values = [pf[key] for pf in per_frame]
        aggregated[key] = {
            "mean": round(stats.fmean(values), 4),
            "variance": round(stats.pvariance(values), 4),
        }

    return {
        "video_path": video_path,
        "sample_rate_fps": sample_rate_fps,
        "num_frames_sampled": len(frames),
        "per_frame": per_frame,
        "aggregated": aggregated,
    }


def main():
    parser = argparse.ArgumentParser(description="Score a video's aesthetic quality (per-frame + aggregated).")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sample rate in frames/second (default: 1.0)")
    parser.add_argument("--output", type=str, default=None, help="Write JSON output to this path instead of stdout")
    args = parser.parse_args()

    result = score_video(args.video, sample_rate_fps=args.fps)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
