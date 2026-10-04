import unittest

import json
import os
import tempfile

from mongo_util import describe_mongo_url_problem
from train import (
    empty_dataset_hint,
    is_mediapipe_training_doc,
    labeled_feature_row,
    load_dataset_from_json,
    parse_args,
    rows_from_docs,
    session_features,
)


class TrainDatasetTests(unittest.TestCase):
    def test_identifies_mediapipe_docs(self):
        self.assertTrue(is_mediapipe_training_doc({"engine": "mediapipe"}))
        self.assertTrue(is_mediapipe_training_doc({"source_label": "mediapipe｜相機（即時）"}))
        self.assertFalse(is_mediapipe_training_doc({"engine": "pose", "source_label": "相機（即時）"}))

    def test_session_features_length(self):
        doc = {
            "label": "good",
            "frames": [
                {
                    "nodes": [
                        {"joint": "nose", "x": 0.5, "y": 0.2, "z": 0.0},
                        {"joint": "left_hip", "x": 0.4, "y": 0.6, "z": 0.1},
                    ]
                }
            ],
        }
        feats = session_features(doc)
        self.assertIsNotNone(feats)
        self.assertEqual(len(feats), 210)

    def test_rows_skip_unlabeled_and_keep_good_bad(self):
        docs = [
            {"_id": "1", "label": None, "frames": [{"nodes": [{"joint": "nose", "x": 0, "y": 0, "z": 0}]}]},
            {"_id": "2", "label": "good", "frames": [{"nodes": [{"joint": "nose", "x": 0.1, "y": 0.2, "z": 0.3}]}]},
            {"_id": "3", "label": "bad", "frames": [{"nodes": [{"joint": "nose", "x": 0.2, "y": 0.3, "z": 0.4}]}]},
        ]
        X, y = rows_from_docs(docs)
        self.assertEqual(len(X), 2)
        self.assertEqual(sorted(y.tolist()), [0, 1])

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
                {
                    "label": "good",
                    "engine": "mediapipe",
                    "frames": [{"nodes": [{"joint": "nose", "x": 0.1, "y": 0.2, "z": 0.3}]}],
                },
                {
                    "label": "bad",
                    "engine": "mediapipe",
                    "frames": [{"nodes": [{"joint": "nose", "x": 0.2, "y": 0.3, "z": 0.4}]}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "export.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            X, y = load_dataset_from_json(path, "mediapipe")
        self.assertEqual(len(X), 2)
        self.assertEqual(sorted(y.tolist()), [0, 1])

    def test_parse_from_json_args(self):
        args = parse_args(["--engine", "mediapipe", "--from-json", "export.json"])
        self.assertEqual(args.from_json, "export.json")

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
