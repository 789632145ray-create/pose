import json
import math
import os
import tempfile
import unittest

from mongo_util import describe_mongo_url_problem
from train import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    FEATURE_COLUMNS,
    JOINTS,
    empty_dataset_hint,
    frame_windows,
    horizontally_flip_frames,
    is_mediapipe_training_doc,
    labeled_feature_row,
    load_dataset_from_json,
    parse_args,
    rows_from_docs,
    session_features,
)


def _core_frame(
    *,
    hip_x: float = 0.50,
    hip_y: float = 0.60,
    shoulder_y: float = 0.35,
    lean_x: float = 0.0,
    visibility: float = 0.9,
) -> dict:
    """組一幀含核心關節的假資料（約略正面站姿）。"""
    shoulder_mid = (hip_x + lean_x, shoulder_y)
    hip_mid = (hip_x, hip_y)
    nodes = [
        {"joint": "shoulder_mid", "x": shoulder_mid[0], "y": shoulder_mid[1], "z": 0.0, "visibility": visibility},
        {"joint": "hip_mid", "x": hip_mid[0], "y": hip_mid[1], "z": 0.0, "visibility": visibility},
        {"joint": "left_shoulder", "x": hip_x - 0.12 + lean_x, "y": shoulder_y, "z": 0.0, "visibility": visibility},
        {"joint": "right_shoulder", "x": hip_x + 0.12 + lean_x, "y": shoulder_y, "z": 0.0, "visibility": visibility},
        {"joint": "left_elbow", "x": hip_x - 0.16, "y": 0.45, "z": 0.0, "visibility": visibility},
        {"joint": "right_elbow", "x": hip_x + 0.16, "y": 0.45, "z": 0.0, "visibility": visibility},
        {"joint": "left_wrist", "x": hip_x - 0.18, "y": 0.55, "z": 0.0, "visibility": visibility},
        {"joint": "right_wrist", "x": hip_x + 0.18, "y": 0.55, "z": 0.0, "visibility": visibility},
        {"joint": "left_hip", "x": hip_x - 0.08, "y": hip_y, "z": 0.0, "visibility": visibility},
        {"joint": "right_hip", "x": hip_x + 0.08, "y": hip_y, "z": 0.0, "visibility": visibility},
        {"joint": "left_knee", "x": hip_x - 0.09, "y": 0.75, "z": 0.0, "visibility": visibility},
        {"joint": "right_knee", "x": hip_x + 0.09, "y": 0.75, "z": 0.0, "visibility": visibility},
        {"joint": "left_ankle", "x": hip_x - 0.09, "y": 0.90, "z": 0.0, "visibility": visibility},
        {"joint": "right_ankle", "x": hip_x + 0.09, "y": 0.90, "z": 0.0, "visibility": visibility},
        {"joint": "left_heel", "x": hip_x - 0.10, "y": 0.92, "z": 0.0, "visibility": visibility},
        {"joint": "right_heel", "x": hip_x + 0.10, "y": 0.92, "z": 0.0, "visibility": visibility},
        {"joint": "left_foot_index", "x": hip_x - 0.08, "y": 0.94, "z": 0.0, "visibility": visibility},
        {"joint": "right_foot_index", "x": hip_x + 0.08, "y": 0.94, "z": 0.0, "visibility": visibility},
        # 雜訊關節：應被特徵工程忽略
        {"joint": "nose", "x": 0.5, "y": 0.1, "z": 0.0, "visibility": visibility},
        {"joint": "left_eye", "x": 0.48, "y": 0.08, "z": 0.0, "visibility": visibility},
        {"joint": "left_pinky", "x": 0.2, "y": 0.5, "z": 0.0, "visibility": visibility},
    ]
    return {"nodes": nodes}


def _session(label: str, n_frames: int = 40, session_id: str = "s1", **frame_kw) -> dict:
    return {
        "_id": session_id,
        "label": label,
        "engine": "mediapipe",
        "frames": [_core_frame(**frame_kw) for _ in range(n_frames)],
    }


class TrainDatasetTests(unittest.TestCase):
    def test_identifies_mediapipe_docs(self):
        self.assertTrue(is_mediapipe_training_doc({"engine": "mediapipe"}))
        self.assertTrue(is_mediapipe_training_doc({"source_label": "mediapipe｜相機（即時）"}))
        self.assertFalse(is_mediapipe_training_doc({"engine": "pose", "source_label": "相機（即時）"}))

    def test_feature_columns_exclude_face_and_fingers(self):
        joined = ",".join(FEATURE_COLUMNS)
        self.assertNotIn("eye", joined)
        self.assertNotIn("mouth", joined)
        self.assertNotIn("ear", joined)
        self.assertNotIn("pinky", joined)
        self.assertNotIn("thumb", joined)
        self.assertIn("left_knee", joined)
        self.assertIn("torso_lean_mean", joined)
        self.assertIn("pelvis_tilt_max", joined)
        self.assertEqual(len(JOINTS), 18)
        # 18 joints × mean/std × x/y + 6 angles × mean/std/min/max
        self.assertEqual(len(FEATURE_COLUMNS), 18 * 4 + 6 * 4)

    def test_session_features_length_and_ignores_low_visibility(self):
        good = {"frames": [_core_frame(visibility=0.9)]}
        feats = session_features(good)
        self.assertIsNotNone(feats)
        self.assertEqual(len(feats), len(FEATURE_COLUMNS))

        invisible = {"frames": [_core_frame(visibility=0.1)]}
        self.assertIsNone(session_features(invisible))

    def test_session_features_normalizes_away_global_translation(self):
        a = session_features({"frames": [_core_frame(hip_x=0.40, hip_y=0.55)]})
        b = session_features({"frames": [_core_frame(hip_x=0.60, hip_y=0.70)]})
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertTrue(math.isclose(a[0], b[0], abs_tol=1e-6))  # hip_mid mean_x ~ 0
        self.assertTrue(math.isclose(a[2], b[2], abs_tol=1e-6))  # hip_mid mean_y ~ 0

    def test_rows_windowing_expands_samples_with_groups(self):
        docs = [
            _session("good", n_frames=40, session_id="g1"),
            _session("bad", n_frames=40, session_id="b1", lean_x=0.05),
            {"_id": "skip", "label": None, "frames": [_core_frame()]},
        ]
        X, y, groups = rows_from_docs(docs, window_size=30, stride=15)
        # 40 幀 → windows at 0 and 10 (+ tail if needed); both sessions labeled
        self.assertGreaterEqual(len(X), 4)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(y), len(groups))
        self.assertEqual(set(groups.tolist()), {"g1", "b1"})
        self.assertEqual(sorted(set(y.tolist())), [0, 1])
        self.assertEqual(X.shape[1], len(FEATURE_COLUMNS))

    def test_frame_windows_matches_predict_defaults(self):
        frames = list(range(60))
        windows = frame_windows(frames, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_WINDOW_STRIDE)
        self.assertEqual(DEFAULT_WINDOW_SIZE, 30)
        self.assertEqual(DEFAULT_WINDOW_STRIDE, 15)
        self.assertEqual(len(windows[0]), 30)
        self.assertEqual([w[0] for w in windows], [0, 15, 30])

    def test_horizontal_flip_swaps_sides_and_mirrors_x(self):
        frame = _core_frame()
        flipped = horizontally_flip_frames([frame])[0]
        joints = {n["joint"]: n for n in flipped["nodes"]}
        self.assertIn("right_hip", joints)
        original_left = next(n for n in frame["nodes"] if n["joint"] == "left_hip")
        self.assertAlmostEqual(joints["right_hip"]["x"], 1.0 - original_left["x"])

    def test_labeled_feature_row_rejects_empty_frames(self):
        self.assertIsNone(labeled_feature_row({"label": "good", "frames": []}))

    def test_empty_dataset_hint_mentions_localhost_vs_railway(self):
        hint = empty_dataset_hint("mediapipe", "mongodb://localhost:27017", from_api=False)
        self.assertIn("localhost", hint)
        self.assertIn("POSE_MONGO_URL", hint)
        self.assertIn("--from-json", hint)

    def test_placeholder_mongo_url_is_rejected(self):
        self.assertIsNotNone(describe_mongo_url_problem("mongodb+srv://..."))
        self.assertIsNotNone(describe_mongo_url_problem("mongodb+srv://user:pass@cluster..mongodb.net"))
        self.assertIsNone(describe_mongo_url_problem("mongodb://localhost:27017"))
        self.assertIsNone(
            describe_mongo_url_problem("mongodb+srv://user:pass@cluster0.abcd123.mongodb.net/?retryWrites=true")
        )

    def test_load_dataset_from_json(self):
        payload = {
            "engine": "mediapipe",
            "sessions": [
                _session("good", n_frames=35, session_id="a"),
                _session("bad", n_frames=35, session_id="b", lean_x=0.04),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "export.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            X, y, groups = load_dataset_from_json(path, "mediapipe")
        self.assertGreaterEqual(len(X), 2)
        self.assertEqual(sorted(set(y.tolist())), [0, 1])
        self.assertEqual(X.shape[1], len(FEATURE_COLUMNS))
        self.assertEqual(len(groups), len(X))

    def test_parse_from_json_args(self):
        args = parse_args(["--engine", "mediapipe", "--from-json", "export.json"])
        self.assertEqual(args.from_json, "export.json")
        self.assertEqual(args.window_size, DEFAULT_WINDOW_SIZE)
        self.assertEqual(args.window_stride, DEFAULT_WINDOW_STRIDE)

    def test_parse_from_api_args(self):
        args = parse_args(
            [
                "--engine",
                "mediapipe",
                "--from-api",
                "https://runpose-backend-production.up.railway.app",
                "--username",
                "tester",
            ]
        )
        self.assertEqual(args.engine, "mediapipe")
        self.assertTrue(args.from_api.endswith(".railway.app"))
        self.assertEqual(args.username, "tester")


if __name__ == "__main__":
    unittest.main()
