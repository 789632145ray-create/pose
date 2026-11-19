import unittest

from side_view_preprocessor import FEATURE_COLUMNS, SideViewPreprocessor


def _node(joint: str, x: float, y: float, z: float = 0.0) -> dict:
    return {
        "joint": joint,
        "x": x,
        "y": y,
        "z": z,
        "visibility": 1.0,
        "presence": 1.0,
    }


def _right_facing_frame(shift: float = 0.0) -> list[dict]:
    # 朝右：鼻／腳尖 X > 腳跟／髖；近側為右側肢體。
    return [
        _node("nose", 0.62 + shift, 0.20),
        _node("right_ear", 0.58 + shift, 0.22),
        _node("right_shoulder", 0.55 + shift, 0.35),
        _node("right_hip", 0.50 + shift, 0.55),
        _node("right_knee", 0.54 + shift, 0.72),
        _node("right_ankle", 0.58 + shift, 0.90),
        _node("right_heel", 0.56 + shift, 0.92),
        _node("right_foot_index", 0.62 + shift, 0.91),
        _node("left_hip", 0.48 + shift, 0.55),
        _node("left_heel", 0.46 + shift, 0.92),
    ]


class SideViewPreprocessorTests(unittest.TestCase):
    def test_feature_columns_length(self):
        self.assertEqual(len(FEATURE_COLUMNS), 8)

    def test_facing_right_and_vector_length(self):
        prep = SideViewPreprocessor()
        feats = None
        for i in range(6):
            feats = prep.process_frame(_right_facing_frame(shift=0.01 * i))
        self.assertIsNotNone(feats)
        self.assertEqual(feats.facing, "right")
        self.assertEqual(len(feats.feature_vector()), len(FEATURE_COLUMNS))
        self.assertTrue(feats.core_reliable)

    def test_facing_left_flips_to_consistent_near_side(self):
        prep = SideViewPreprocessor()
        # 鏡像朝左樣本：鼻／腳尖在左（較小 X）。
        left_frame = [
            _node("nose", 0.38, 0.20),
            _node("left_ear", 0.42, 0.22),
            _node("left_shoulder", 0.45, 0.35),
            _node("left_hip", 0.50, 0.55),
            _node("left_knee", 0.46, 0.72),
            _node("left_ankle", 0.42, 0.90),
            _node("left_heel", 0.44, 0.92),
            _node("left_foot_index", 0.38, 0.91),
            _node("right_hip", 0.52, 0.55),
            _node("right_heel", 0.54, 0.92),
        ]
        feats = None
        for i in range(6):
            shifted = [{**n, "x": float(n["x"]) - 0.01 * i} for n in left_frame]
            feats = prep.process_frame(shifted)
        self.assertIsNotNone(feats)
        self.assertEqual(feats.near_side, "right")
        self.assertIsNotNone(feats.trunk_forward_lean_deg)


if __name__ == "__main__":
    unittest.main()
