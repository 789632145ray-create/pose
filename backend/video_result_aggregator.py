"""
影片級「好／壞」後處理聚合（VideoResultAggregator）。

將多筆時間點預測收斂成單一 final_label + final_confidence，供 App 直接渲染。
訓練／推論後處理請共用此模組，避免各端各自算分造成跳動。

步驟：
  1) 過濾低信心雜訊（min_conf，可調；過濾後若空則退回全部）
  2) 信心加權投票決定 final_label
  3) 勝出類別的代表性信心：Top-K 平均（預設）或 Peak
  4) 輸出統一 JSON 結構
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Sequence


class ScoreMode(str, Enum):
    TOP_K = "topk"  # 模式 A：勝出類別前 20%（至少 1、最多 5）平均
    PEAK = "peak"  # 模式 B：勝出類別單筆 max(conf)


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


class VideoResultAggregator:
    """影片級預測聚合器。"""

    def __init__(
        self,
        min_conf: float = 0.5,
        score_mode: ScoreMode | str = ScoreMode.TOP_K,
    ) -> None:
        if not 0.0 <= float(min_conf) <= 1.0:
            raise ValueError("min_conf 必須在 0~1")
        self.min_conf = float(min_conf)
        self.score_mode = ScoreMode(score_mode)

    def aggregate(self, predictions: Iterable[FramePrediction | dict[str, Any]]) -> VideoAggregateResult:
        samples = [self._coerce(p) for p in predictions]
        if not samples:
            return VideoAggregateResult(
                final_label="bad",
                final_confidence=0.0,
                dominance_ratio=0.0,
                valid_samples_count=0,
            )

        # 1) 過濾低信心；若全滅則退回全部原始預測。
        filtered = [p for p in samples if p.confidence >= self.min_conf]
        used = filtered if filtered else list(samples)

        # 2) 信心加權投票。
        sum_good = sum(p.confidence for p in used if p.label == "good")
        sum_bad = sum(p.confidence for p in used if p.label == "bad")
        total = sum_good + sum_bad
        if total <= 0:
            # 理論上不會發生（confidence>0）；保底。
            final_label = "good" if sum(1 for p in used if p.label == "good") >= len(used) / 2 else "bad"
            dominance = 0.0
        elif sum_good >= sum_bad:
            final_label = "good"
            dominance = sum_good / total
        else:
            final_label = "bad"
            dominance = sum_bad / total

        winner_confs = sorted(
            (p.confidence for p in used if p.label == final_label),
            reverse=True,
        )
        if not winner_confs:
            # 過濾後某一類可能為空但總分比較仍選到它（不應）；退回對側或 0。
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
            # 若只給 probability_good，bad 信心 = 1 - p_good
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
