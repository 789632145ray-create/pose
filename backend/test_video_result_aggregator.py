import unittest

from video_result_aggregator import (
    FramePrediction,
    ScoreMode,
    VideoResultAggregator,
    _topk_count,
)


class VideoResultAggregatorTests(unittest.TestCase):
    def test_topk_count_bounds(self):
        self.assertEqual(_topk_count(0), 0)
        self.assertEqual(_topk_count(1), 1)
        self.assertEqual(_topk_count(4), 1)  # 20% → 1
        self.assertEqual(_topk_count(10), 2)
        self.assertEqual(_topk_count(100), 5)  # capped at 5

    def test_filters_low_confidence_then_weighted_vote(self):
        preds = [
            FramePrediction("good", 0.9),
            FramePrediction("good", 0.85),
            FramePrediction("bad", 0.95),
            FramePrediction("bad", 0.2),  # filtered when min_conf=0.5
            FramePrediction("good", 0.4),  # filtered
        ]
        result = VideoResultAggregator(min_conf=0.5).aggregate(preds)
        # used: good 0.9+0.85=1.75, bad 0.95 → good wins
        self.assertEqual(result.final_label, "good")
        self.assertEqual(result.valid_samples_count, 3)
        self.assertAlmostEqual(result.dominance_ratio, 1.75 / (1.75 + 0.95), places=5)
        # Top-K of 2 goods → 20% of 2 = 1 → mean of top1 = 0.9
        self.assertAlmostEqual(result.final_confidence, 0.9, places=5)

    def test_fallback_when_all_filtered(self):
        preds = [
            FramePrediction("bad", 0.3),
            FramePrediction("bad", 0.4),
            FramePrediction("good", 0.2),
        ]
        result = VideoResultAggregator(min_conf=0.5).aggregate(preds)
        self.assertEqual(result.valid_samples_count, 3)
        self.assertEqual(result.final_label, "bad")

    def test_peak_mode(self):
        preds = [
            FramePrediction("good", 0.7),
            FramePrediction("good", 0.95),
            FramePrediction("bad", 0.6),
        ]
        result = VideoResultAggregator(min_conf=0.5, score_mode=ScoreMode.PEAK).aggregate(preds)
        self.assertEqual(result.final_label, "good")
        self.assertAlmostEqual(result.final_confidence, 0.95, places=5)

    def test_dict_input_probability_good(self):
        result = VideoResultAggregator(min_conf=0.5).aggregate(
            [
                {"label": "good", "probability_good": 0.8},
                {"label": "bad", "probability_good": 0.2},  # bad conf = 0.8
            ]
        )
        self.assertIn(result.final_label, {"good", "bad"})
        self.assertEqual(result.valid_samples_count, 2)

    def test_empty(self):
        result = VideoResultAggregator().aggregate([])
        self.assertEqual(result.valid_samples_count, 0)
        self.assertEqual(result.final_confidence, 0.0)

    def test_drops_ambiguous_mid_band(self):
        preds = [
            FramePrediction("good", 0.92),
            FramePrediction("good", 0.88),
            FramePrediction("good", 0.50),  # 模糊區間，應排除
            FramePrediction("bad", 0.48),  # 模糊區間（且 < min_conf）
            FramePrediction("bad", 0.70),
        ]
        agg = VideoResultAggregator(min_conf=0.5)
        used = agg.clear_windows(preds)
        confs = sorted(p.confidence for p in used)
        self.assertNotIn(0.50, confs)
        self.assertNotIn(0.48, confs)
        result = agg.aggregate(preds)
        self.assertEqual(result.final_label, "good")
        # 勝出 good 的 top 60% → 兩筆都留下，topk 取 0.92
        self.assertGreaterEqual(result.final_confidence, 0.85)

    def test_all_ambiguous_falls_back_then_scores_top_frac(self):
        preds = [
            FramePrediction("good", 0.52),
            FramePrediction("bad", 0.47),
            FramePrediction("good", 0.49),
            FramePrediction("bad", 0.51),
            FramePrediction("good", 0.46),
        ]
        agg = VideoResultAggregator(min_conf=0.45, top_frac=0.6)
        used = agg.clear_windows(preds)
        # 全模糊 → 投票退回全部 5 筆
        self.assertEqual(len(used), 5)
        result = agg.aggregate(preds)
        self.assertEqual(result.valid_samples_count, 5)
        # 分數只取勝出類別前 60%
        self.assertGreater(result.final_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
