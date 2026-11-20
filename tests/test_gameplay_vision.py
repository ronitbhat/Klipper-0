import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gameplay_vision as vision


VALID_RESPONSE = {
    "event_type": "vehicle_stunt",
    "notable": True,
    "observations": ["Vehicle is visibly upside down in mid-air."],
    "uncertainties": ["The landing is not visible."],
}


class GameplayVisionTests(unittest.TestCase):
    def test_valid_response_is_accepted(self):
        result = vision._validate_response(VALID_RESPONSE)
        self.assertEqual(result["event_type"], "vehicle_stunt")
        self.assertTrue(result["notable"])

    def test_invalid_event_type_is_rejected(self):
        data = {**VALID_RESPONSE, "event_type": "made_up_event"}
        with self.assertRaises(ValueError):
            vision._validate_response(data)

    def test_incomplete_json_is_rejected(self):
        with self.assertRaises(ValueError):
            vision._parse_response('{"event_type": "vehicle_stunt"')

    def test_invalid_json_is_rejected(self):
        with self.assertRaises(ValueError):
            vision._parse_response("not json")

    def test_too_many_observations_are_rejected(self):
        data = {
            **VALID_RESPONSE,
            "observations": ["one", "two", "three", "four"],
        }
        with self.assertRaises(ValueError):
            vision._validate_response(data)

    def test_observation_over_100_characters_is_rejected(self):
        data = {
            **VALID_RESPONSE,
            "observations": ["x" * 101],
        }
        with self.assertRaises(ValueError):
            vision._validate_response(data)

    def test_analysis_retries_with_fewer_frames(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            def fake_extract(video_path, timestamp, output_path):
                output_path.write_bytes(b"fake frame")

            with (
                patch.object(vision, "_extract_frame", side_effect=fake_extract),
                patch.object(
                    vision,
                    "_analyze_frames",
                    side_effect=[RuntimeError("GPU memory error"), VALID_RESPONSE],
                ) as analyze,
            ):
                result = vision.analyze_video_window(
                    video.name, 0, 10, frame_count=3
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(analyze.call_count, 2)

    def test_analysis_returns_error_after_all_attempts_fail(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            def fake_extract(video_path, timestamp, output_path):
                output_path.write_bytes(b"fake frame")

            with (
                patch.object(vision, "_extract_frame", side_effect=fake_extract),
                patch.object(
                    vision,
                    "_analyze_frames",
                    side_effect=RuntimeError("GPU memory error"),
                ),
            ):
                result = vision.analyze_video_window(
                    video.name, 0, 10, frame_count=3
                )

        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["analysis"])
        self.assertEqual(len(result["attempts"]), 3)


    def test_frame_extraction_failure_does_not_retry(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with (
                patch.object(
                    vision,
                    "_extract_frame",
                    side_effect=RuntimeError("FFmpeg extraction failed"),
                ) as extract,
                patch.object(vision, "_analyze_frames") as analyze,
            ):
                result = vision.analyze_video_window(
                    video.name, 0, 10, frame_count=3
                )

        self.assertEqual(result["status"], "error")
        self.assertIn("FFmpeg extraction failed", result["error"])
        self.assertEqual(extract.call_count, 1)
        analyze.assert_not_called()

    def test_qwen_timeout_does_not_retry(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            def fake_extract(video_path, timestamp, output_path):
                output_path.write_bytes(b"fake frame")

            with (
                patch.object(vision, "_extract_frame", side_effect=fake_extract),
                patch.object(
                    vision,
                    "_analyze_frames",
                    side_effect=subprocess.TimeoutExpired(
                        cmd="llama-mtmd-cli",
                        timeout=180,
                    ),
                ) as analyze,
            ):
                result = vision.analyze_video_window(
                    video.name, 0, 10, frame_count=3
                )

        self.assertEqual(result["status"], "error")
        self.assertIn("timed out", result["error"])
        self.assertEqual(analyze.call_count, 1)

    def test_unexpected_response_fields_are_rejected(self):
        data = {**VALID_RESPONSE, "extra": "unexpected"}
        with self.assertRaises(ValueError):
            vision._validate_response(data)

    def test_missing_response_fields_are_rejected(self):
        data = {
            key: value
            for key, value in VALID_RESPONSE.items()
            if key != "uncertainties"
        }
        with self.assertRaises(ValueError):
            vision._validate_response(data)

    def test_empty_observation_is_rejected(self):
        data = {**VALID_RESPONSE, "observations": ["   "]}
        with self.assertRaises(ValueError):
            vision._validate_response(data)

    def test_boolean_frame_count_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with self.assertRaises(ValueError):
                vision.analyze_video_window(
                    video.name, 0, 10, frame_count=True
                )

    def test_infinite_timestamp_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with self.assertRaises(ValueError):
                vision.analyze_video_window(
                    video.name, 0, float("inf")
                )

    def test_nan_timestamp_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with self.assertRaises(ValueError):
                vision.analyze_video_window(
                    video.name, 0, float("nan")
                )



if __name__ == "__main__":
    unittest.main()
