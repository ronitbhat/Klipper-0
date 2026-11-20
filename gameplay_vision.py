
"""Standalone GTA gameplay analysis using local Qwen3-VL."""

import json
import subprocess
import tempfile
import math
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"

EVENT_TYPES = {
    "vehicle_stunt",
    "crash",
    "police_chase",
    "explosion",
    "combat",
    "npc_interaction",
    "mission_event",
    "ordinary_gameplay",
    "uncertain",
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "enum": sorted(EVENT_TYPES),
        },
        "notable": {"type": "boolean"},
        "observations": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 3,
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 2,
        },
    },
    "required": [
        "event_type",
        "notable",
        "observations",
        "uncertainties",
    ],
}


def _extract_frame(video_path, timestamp, output_path):
    """Extract one frame at a specific video timestamp."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale=640:-2",
        "-q:v", "3",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"FFmpeg could not extract frame at {timestamp:.2f}s: "
            f"{result.stderr[-1000:]}"
        )



def _validate_response(data):
    """Validate Qwen's structured response without external dependencies."""
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object.")

    expected_fields = {
        "event_type",
        "notable",
        "observations",
        "uncertainties",
    }

    if set(data) != expected_fields:
        raise ValueError("Response contains missing or unexpected fields.")

    if data["event_type"] not in EVENT_TYPES:
        raise ValueError("Invalid event_type.")

    if not isinstance(data["notable"], bool):
        raise ValueError("notable must be a boolean.")

    for field, limit in (("observations", 3), ("uncertainties", 2)):
        values = data[field]

        if not isinstance(values, list) or len(values) > limit:
            raise ValueError(
                f"{field} must be a list of at most {limit} items."
            )

        if not all(
            isinstance(item, str)
            and 0 < len(item.strip()) <= 100
            for item in values
        ):
            raise ValueError(f"Invalid entries in {field}.")

    return {
        "event_type": data["event_type"],
        "notable": data["notable"],
        "observations": data["observations"],
        "uncertainties": data["uncertainties"],
    }



def _parse_response(output):
    """Parse JSON, tolerating harmless text surrounding the object."""
    output = output.strip()
    start = output.find("{")
    end = output.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("Qwen did not return a complete JSON object.")

    data = json.loads(output[start:end + 1])
    return _validate_response(data)


def _sample_timestamps(start, end, frame_count):
    fractions = [
        (i + 1) / (frame_count + 1)
        for i in range(frame_count)
    ]
    return [
        start + (end - start) * fraction
        for fraction in fractions
    ]


def _analyze_frames(frame_paths, timeout, model):
    """Run Qwen with JSON Schema-constrained generation."""
    frame_list = ",".join(str(path) for path in frame_paths)

    prompt = (
        "Analyze these GTA gameplay frames in chronological order. "
        "Identify the main visible gameplay event. "
        "Report only directly visible facts. "
        "Do not guess vehicle models, causes, or unseen outcomes. "
        "Give at most three short observations and two uncertainties. "
        "If no notable event is clear, use ordinary_gameplay or uncertain. "
        "Return only the requested JSON object."
    )

    command = [
        "llama-mtmd-cli",
        "-hf", model,
        "--gpu-layers", "99",
        "--ctx-size", "4096",
        "--image-min-tokens", "1024",
        "-n", "256",
        "--image", frame_list,
        "-j", json.dumps(OUTPUT_SCHEMA),
        "-p", prompt,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] or "Qwen execution failed.")

    return _parse_response(result.stdout)



def analyze_video_window(
    video_path,
    start,
    end,
    frame_count=3,
    model=DEFAULT_MODEL,
    timeout=180,
):
    """
    Analyze a video interval using local Qwen3-VL.

    Retries Qwen analysis with fewer frames when analysis fails.
    Frame extraction failures and timeouts stop immediately.
    """
    video_path = Path(video_path).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    start = float(start)
    end = float(end)

    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("start and end must be finite numbers.")

    if start < 0 or end <= start:
        raise ValueError("Require 0 <= start < end.")

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("frame_count must be an integer.")

    if not 1 <= frame_count <= 3:
        raise ValueError("frame_count must be between 1 and 3.")

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive number.")

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    errors = []

    try:
        with tempfile.TemporaryDirectory(prefix="gta-vision-") as temp:
            temp_dir = Path(temp)

            for count in range(frame_count, 0, -1):
                timestamps = _sample_timestamps(start, end, count)
                frame_paths = []

                # Frame extraction is not retried. If FFmpeg fails,
                # fewer frames will not solve the underlying problem.
                for index, timestamp in enumerate(timestamps, start=1):
                    frame_path = temp_dir / f"frame-{index:02d}.jpg"

                    _extract_frame(
                        video_path,
                        timestamp,
                        frame_path,
                    )

                    frame_paths.append(frame_path)

                try:
                    analysis = _analyze_frames(
                        frame_paths,
                        timeout,
                        model,
                    )

                    return {
                        "status": "success",
                        "analysis": analysis,
                        "timestamps": timestamps,
                        "frame_count": count,
                        "video": str(video_path),
                        "window": {
                            "start": start,
                            "end": end,
                        },
                        "model": model,
                        "retry_count": frame_count - count,
                    }

                except subprocess.TimeoutExpired:
                    return {
                        "status": "error",
                        "analysis": None,
                        "timestamps": timestamps,
                        "error": f"Qwen timed out after {timeout} seconds.",
                        "attempts": errors,
                    }

                except (RuntimeError, ValueError) as exc:
                    errors.append({
                        "frames": count,
                        "error": str(exc),
                    })

            return {
                "status": "error",
                "analysis": None,
                "timestamps": timestamps,
                "error": "All Qwen analysis attempts failed.",
                "attempts": errors,
            }

    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "error",
            "analysis": None,
            "error": str(exc),
        }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze GTA gameplay with local Qwen3-VL."
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    result = analyze_video_window(
        args.video,
        args.start,
        args.end,
        frame_count=args.frames,
        timeout=args.timeout,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
