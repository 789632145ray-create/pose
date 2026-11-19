"""
側面視角前處理器（SideViewPreprocessor）—— 與 iOS `SideViewPreprocessor.swift` 對齊。

訓練與離線推論必須呼叫同一套步驟，避免前處理落差：
  1) 朝向判斷＋水平翻轉（統一朝右，並對調 left/right 標籤）
  2) 近側肢體篩選
  3) 中值濾波＋膝角運動學約束
  4) 近側髖原點＋軀幹尺度正規化
  5) 生物力學特徵向量（固定欄位順序）

特徵欄位（與 Swift `SideViewFrameFeatures.featureColumnNames` 一致）：
  trunk_forward_lean_deg, forward_head_deg, knee_flexion_deg, hip_flexion_deg,
  overstride_norm, vertical_oscillation_norm, core_reliable, facing_sign
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

FEATURE_COLUMNS = [
    "trunk_forward_lean_deg",
    "forward_head_deg",
    "knee_flexion_deg",
    "hip_flexion_deg",
    "overstride_norm",
    "vertical_oscillation_norm",
    "core_reliable",
    "facing_sign",
]

NEAR_SUFFIXES = [
    "ear",
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
    "heel",
    "foot_index",
]


def _avg(a: float | None, b: float | None) -> float | None:
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def _angle_deg(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if mag < 1e-9:
        return 180.0
    cos = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / mag))
    return math.degrees(math.acos(cos))


def _swap_side(joint: str) -> str:
    if joint.startswith("left_"):
        return "right_" + joint[len("left_") :]
    if joint.startswith("right_"):
        return "left_" + joint[len("right_") :]
    return joint


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


@dataclass
class SideViewFrameFeatures:
    facing: str = "unknown"
    near_side: str = "right"
    trunk_forward_lean_deg: float | None = None
    forward_head_deg: float | None = None
    knee_flexion_deg: float | None = None
    hip_flexion_deg: float | None = None
    overstride_norm: float | None = None
    vertical_oscillation_norm: float | None = None
    core_reliable: bool = False

    def feature_vector(self) -> list[float]:
        facing_sign = 1.0 if self.facing == "right" else (-1.0 if self.facing == "left" else 0.0)
        return [
            self.trunk_forward_lean_deg or 0.0,
            self.forward_head_deg or 0.0,
            self.knee_flexion_deg or 0.0,
            self.hip_flexion_deg or 0.0,
            self.overstride_norm or 0.0,
            self.vertical_oscillation_norm or 0.0,
            1.0 if self.core_reliable else 0.0,
            facing_sign,
        ]


@dataclass
class SideViewPreprocessor:
    """與 App 端同一個前處理契約；train／offline infer 請只透過此類。"""

    median_window: int = 5
    max_knee_angle_delta_deg: float = 45.0
    history: list[dict[str, dict[str, float]]] = field(default_factory=list)
    last_trusted: dict[str, dict[str, float]] = field(default_factory=dict)
    last_knee_flexion_deg: float | None = None
    recent_hip_y: list[float] = field(default_factory=list)
    recent_ankle_y: list[float] = field(default_factory=list)
    pelvis_x_history: list[float] = field(default_factory=list)
    last_facing: str = "unknown"

    def reset(self) -> None:
        self.history.clear()
        self.last_trusted.clear()
        self.last_knee_flexion_deg = None
        self.recent_hip_y.clear()
        self.recent_ankle_y.clear()
        self.pelvis_x_history.clear()
        self.last_facing = "unknown"

    def process_frame(self, nodes: list[dict[str, Any]]) -> SideViewFrameFeatures:
        frame = {str(n["joint"]): n for n in nodes if "joint" in n}
        facing = self._detect_facing(frame)
        if facing != "unknown":
            self.last_facing = facing
        oriented = self._normalize_orientation(frame, self.last_facing)
        near = self._select_near_side(oriented, "right")
        smoothed = self._median_filter(near)
        constrained = self._kinematic(smoothed)
        return self._extract(constrained, self.last_facing, "right")

    def process_session(self, frames: list[dict[str, Any]]) -> list[list[float]]:
        """整段 session → 每幀特徵向量（訓練用）。"""
        self.reset()
        rows: list[list[float]] = []
        for frame in frames:
            nodes = frame.get("nodes") or []
            rows.append(self.process_frame(nodes).feature_vector())
        return rows

    def _detect_facing(self, frame: dict[str, dict[str, Any]]) -> str:
        votes_r = 0
        votes_l = 0

        def vote(front: float | None, back: float | None) -> None:
            nonlocal votes_r, votes_l
            if front is None or back is None:
                return
            if front > back:
                votes_r += 1
            elif front < back:
                votes_l += 1

        def x(name: str) -> float | None:
            node = frame.get(name)
            return float(node["x"]) if node is not None and "x" in node else None

        nose = x("nose")
        ear = _avg(x("left_ear"), x("right_ear"))
        heel = _avg(x("left_heel"), x("right_heel"))
        toe = _avg(x("left_foot_index"), x("right_foot_index"))
        hip = _avg(x("left_hip"), x("right_hip"))
        vote(nose, heel)
        vote(ear, heel)
        vote(toe, heel)
        vote(nose, hip)
        vote(toe, hip)
        if hip is not None:
            self.pelvis_x_history.append(hip)
            self.pelvis_x_history = self.pelvis_x_history[-8:]
            if len(self.pelvis_x_history) >= 3:
                dx = self.pelvis_x_history[-1] - self.pelvis_x_history[0]
                if dx > 0.01:
                    votes_r += 2
                elif dx < -0.01:
                    votes_l += 2
        if votes_r == votes_l:
            return "unknown"
        return "right" if votes_r > votes_l else "left"

    def _normalize_orientation(self, frame: dict[str, dict[str, Any]], facing: str) -> dict[str, dict[str, Any]]:
        if facing != "left":
            return frame
        out: dict[str, dict[str, Any]] = {}
        for joint, node in frame.items():
            swapped = _swap_side(joint)
            out[swapped] = {
                "joint": swapped,
                "x": 1.0 - float(node["x"]),
                "y": float(node["y"]),
                "z": -float(node.get("z", 0.0)),
                "visibility": float(node.get("visibility", 1.0)),
                "presence": float(node.get("presence", 1.0)),
            }
        return out

    def _select_near_side(self, frame: dict[str, dict[str, Any]], near_side: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key in ("nose", "shoulder_mid", "hip_mid"):
            if key in frame:
                out[key] = frame[key]
        for suffix in NEAR_SUFFIXES:
            key = f"{near_side}_{suffix}"
            if key in frame:
                out[key] = frame[key]
                self.last_trusted[key] = frame[key]
            elif key in self.last_trusted:
                out[key] = self.last_trusted[key]
        return out

    def _median_filter(self, frame: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        self.history.append(frame)
        w = max(3, min(5, self.median_window))
        if len(self.history) > w:
            self.history = self.history[-w:]
        joints = set().union(*(f.keys() for f in self.history))
        out: dict[str, dict[str, Any]] = {}
        for joint in joints:
            samples = [f[joint] for f in self.history if joint in f]
            if not samples:
                continue
            out[joint] = {
                "joint": joint,
                "x": _median([float(s["x"]) for s in samples]),
                "y": _median([float(s["y"]) for s in samples]),
                "z": _median([float(s.get("z", 0.0)) for s in samples]),
                "visibility": _median([float(s.get("visibility", 1.0)) for s in samples]),
                "presence": _median([float(s.get("presence", 1.0)) for s in samples]),
            }
        return out

    def _kinematic(self, frame: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        side = "right"
        hip, knee, ankle = frame.get(f"{side}_hip"), frame.get(f"{side}_knee"), frame.get(f"{side}_ankle")
        if not hip or not knee or not ankle:
            return frame
        angle = _angle_deg(float(hip["x"]), float(hip["y"]), float(knee["x"]), float(knee["y"]), float(ankle["x"]), float(ankle["y"]))
        out = dict(frame)
        if self.last_knee_flexion_deg is not None and abs(angle - self.last_knee_flexion_deg) > self.max_knee_angle_delta_deg:
            for suffix in ("hip", "knee", "ankle", "heel", "foot_index"):
                key = f"{side}_{suffix}"
                if key in self.last_trusted:
                    out[key] = self.last_trusted[key]
            return out
        self.last_knee_flexion_deg = angle
        for suffix in ("hip", "knee", "ankle", "heel", "foot_index"):
            key = f"{side}_{suffix}"
            if key in out:
                self.last_trusted[key] = out[key]
        return out

    def _extract(self, frame: dict[str, dict[str, Any]], facing: str, near_side: str) -> SideViewFrameFeatures:
        hip = frame.get(f"{near_side}_hip")
        if hip is None:
            return SideViewFrameFeatures(facing=facing, near_side=near_side, core_reliable=False)

        shoulder = frame.get(f"{near_side}_shoulder")
        if shoulder is not None:
            scale = max(math.hypot(float(shoulder["x"]) - float(hip["x"]), float(shoulder["y"]) - float(hip["y"])), 1e-6)
        else:
            scale = 1e-6

        def norm(node: dict[str, Any] | None) -> tuple[float, float] | None:
            if node is None:
                return None
            return ((float(node["x"]) - float(hip["x"])) / scale, (float(node["y"]) - float(hip["y"])) / scale)

        n_shoulder = norm(shoulder)
        n_hip = (0.0, 0.0)
        n_knee = norm(frame.get(f"{near_side}_knee"))
        n_ankle = norm(frame.get(f"{near_side}_ankle"))
        n_ear = norm(frame.get(f"{near_side}_ear"))
        n_heel = norm(frame.get(f"{near_side}_heel"))

        trunk = None
        if n_shoulder is not None:
            trunk = math.degrees(math.atan2(n_shoulder[0], -n_shoulder[1]))

        head = None
        if n_ear is not None and n_shoulder is not None:
            head = math.degrees(math.atan2(n_ear[0] - n_shoulder[0], -(n_ear[1] - n_shoulder[1])))

        knee_flex = None
        if n_knee is not None and n_ankle is not None:
            knee_flex = _angle_deg(n_hip[0], n_hip[1], n_knee[0], n_knee[1], n_ankle[0], n_ankle[1])

        hip_flex = None
        if n_shoulder is not None and n_knee is not None:
            hip_flex = _angle_deg(n_shoulder[0], n_shoulder[1], n_hip[0], n_hip[1], n_knee[0], n_knee[1])

        self.recent_hip_y.append(float(hip["y"]) / scale)
        self.recent_hip_y = self.recent_hip_y[-45:]
        vertical = None
        if len(self.recent_hip_y) >= 8:
            vertical = max(self.recent_hip_y) - min(self.recent_hip_y)

        overstride = None
        if n_ankle is not None:
            self.recent_ankle_y.append(n_ankle[1])
            self.recent_ankle_y = self.recent_ankle_y[-9:]
            if len(self.recent_ankle_y) >= 3:
                mid = len(self.recent_ankle_y) // 2
                y_mid = self.recent_ankle_y[mid]
                if y_mid >= self.recent_ankle_y[mid - 1] and y_mid >= self.recent_ankle_y[mid + 1]:
                    overstride = n_heel[0] if n_heel is not None else n_ankle[0]

        core_ok = all(frame.get(f"{near_side}_{s}") is not None for s in ("shoulder", "hip", "knee", "ankle"))
        return SideViewFrameFeatures(
            facing=facing,
            near_side=near_side,
            trunk_forward_lean_deg=trunk,
            forward_head_deg=head,
            knee_flexion_deg=knee_flex,
            hip_flexion_deg=hip_flex,
            overstride_norm=overstride,
            vertical_oscillation_norm=vertical,
            core_reliable=core_ok,
        )
