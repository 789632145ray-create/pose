"""
影片級「好／壞」後處理聚合（VideoResultAggregator）。

將多筆時間點預測收斂成單一 final_label + final_confidence，供 App 直接渲染。
訓練／推論後處理請共用此模組，避免各端各自算分造成跳動。

步驟：
  1) 過濾低信心雜訊（min_conf）；過濾後若空則退回全部
  2) 去掉模糊區間（預設 0.45~0.55）；若全滅則退回上一步（避免片頭／片尾拉低）
  3) 以清晰視窗做信心加權投票 → final_label
  4) 勝出類別再取信心較強的前 top_frac（預設 60%）計算 final_confidence
     （peak 模式仍取最大值；topk 則在強勢子集上再取前 20% cap）
  5) 輸出統一 JSON 結構
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Sequence


class ScoreMode(str, Enum):
    TOP_K = "topk"  # 模式 A：勝出類別前 20%（至少 1、最多 5）平均
    PEAK = "peak"  # 模式 B：勝出類別單筆 max(conf)


DEFAULT_AMBIGUOUS_LOW = 0.45
DEFAULT_AMBIGUOUS_HIGH = 0.55
DEFAULT_TOP_FRAC = 0.60


@dataclass(frozen=True)
class FramePrediction:
    """單一時刻的預測。confidence 為該 label 的信心（0~1）。"""

    label: str  # "good" | "bad"
    confidence: float

    def __post_init__(self) -> None:
        label = str(self.label).strip().lower()
        if label not in {"good", "bad"}:
            raise ValueError(f"label 必須是 good 或 bad，收到：{self.label!r}")
        conf = float(self.confidence)
        if conf < 0 or conf > 1:
            raise ValueError(f"confidence 必須在 0~1，收到：{self.confidence!r}")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", conf)


@dataclass(frozen=True)
class VideoAggregateResult:
    final_label: str
    final_confidence: float
    dominance_ratio: float
    valid_samples_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _topk_count(n: int) -> int:
    """前 20%，至少 1、最多 5。"""
    if n <= 0:
        return 0
    return max(1, min(5, max(1, int(round(n * 0.20)))))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _top_fraction(samples: Sequence[FramePrediction], frac: float) -> list[FramePrediction]:
    """依信心由高到低取前 frac（至少 1 筆）。"""
    if not samples:
        return []
    frac = min(1.0, max(0.0, float(frac)))
    ranked = sorted(samples, key=lambda p: p.confidence, reverse=True)
    k = max(1, int(math.ceil(len(ranked) * frac))) if frac > 0 else len(ranked)
    return list(ranked[:k])


def _in_ambiguous_band(confidence: float, low: float, high: float) -> bool:
    return float(low) <= float(confidence) <= float(high)


class VideoResultAggregator:
    """影片級預測聚合器。"""

    def __init__(
        self,
        min_conf: float = 0.5,
        score_mode: ScoreMode | str = ScoreMode.TOP_K,
        *,
        ambiguous_low: float = DEFAULT_AMBIGUOUS_LOW,
        ambiguous_high: float = DEFAULT_AMBIGUOUS_HIGH,
        top_frac: float = DEFAULT_TOP_FRAC,
    ) -> None:
        if not 0.0 <= float(min_conf) <= 1.0:
            raise ValueError("min_conf 必須在 0~1")
        if not 0.0 <= float(ambiguous_low) <= float(ambiguous_high) <= 1.0:
            raise ValueError("ambiguous_low/high 必須滿足 0 ≤ low ≤ high ≤ 1")
        if not 0.0 < float(top_frac) <= 1.0:
            raise ValueError("top_frac 必須在 (0, 1]")
        self.min_conf = float(min_conf)
        self.score_mode = ScoreMode(score_mode)
        self.ambiguous_low = float(ambiguous_low)
        self.ambiguous_high = float(ambiguous_high)
        self.top_frac = float(top_frac)

    def clear_windows(self, samples: Sequence[FramePrediction]) -> list[FramePrediction]:
        """
        投票用視窗：min_conf 過濾後去掉模糊區間。
        若過濾後全滅 → 退回 min_conf 結果（或原始全部）；
        若再去掉模糊後全滅 → 退回上一步（全部視為模糊時仍可投票）。
        """
        if not samples:
            return []
        filtered = [p for p in samples if p.confidence >= self.min_conf]
        pool = filtered if filtered else list(samples)
        clear = [
            p
            for p in pool
            if not _in_ambiguous_band(p.confidence, self.ambiguous_low, self.ambiguous_high)
        ]
        return clear if clear else pool

    def aggregate(self, predictions: Iterable[FramePrediction | dict[str, Any]]) -> VideoAggregateResult:
        samples = [self._coerce(p) for p in predictions]
        if not samples:
            return VideoAggregateResult(
                final_label="bad",
                final_confidence=0.0,
                dominance_ratio=0.0,
                valid_samples_count=0,
            )

        used = self.clear_windows(samples)

        sum_good = sum(p.confidence for p in used if p.label == "good")
        sum_bad = sum(p.confidence for p in used if p.label == "bad")
        total = sum_good + sum_bad
        if total <= 0:
            final_label = "good" if sum(1 for p in used if p.label == "good") >= len(used) / 2 else "bad"
            dominance = 0.0
        elif sum_good >= sum_bad:
            final_label = "good"
            dominance = sum_good / total
        else:
            final_label = "bad"
            dominance = sum_bad / total

        winner = [p for p in used if p.label == final_label]
        # 分數只用勝出類別中較強的前 top_frac，拉開與模糊視窗的差距。
        strong = _top_fraction(winner, self.top_frac)
        winner_confs = sorted((p.confidence for p in strong), reverse=True)
        if not winner_confs:
            final_confidence = 0.0
        elif self.score_mode == ScoreMode.PEAK:
            final_confidence = float(winner_confs[0])
        else:
            k = _topk_count(len(winner_confs))
            final_confidence = _mean(winner_confs[:k])

        return VideoAggregateResult(
            final_label=final_label,
            final_confidence=float(max(0.0, min(1.0, final_confidence))),
            dominance_ratio=float(max(0.0, min(1.0, dominance))),
            valid_samples_count=len(used),
        )

    @staticmethod
    def _coerce(item: FramePrediction | dict[str, Any]) -> FramePrediction:
        if isinstance(item, FramePrediction):
            return item
        label = str(item.get("label", "")).strip().lower()
        if "confidence" in item:
            conf = float(item["confidence"])
        elif label == "good":
            conf = float(item.get("probability_good", item.get("probabilityGood", 0.0)))
        elif label == "bad":
            if "probability_bad" in item:
                conf = float(item["probability_bad"])
            else:
                conf = 1.0 - float(item.get("probability_good", item.get("probabilityGood", 0.0)))
        else:
            conf = float(item.get("confidence", 0.0))
        return FramePrediction(label=label, confidence=conf)


def predictions_from_proba_rows(
    rows: Sequence[tuple[str, float]],
) -> list[FramePrediction]:
    """[(label, confidence), ...] → FramePrediction 列表。"""
    return [FramePrediction(label=label, confidence=conf) for label, conf in rows]
