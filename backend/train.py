"""
姿勢品質（好 / 壞）監督式學習訓練腳本。

流程：
  1. 從 MongoDB／JSON／API 讀出已標註的姿勢 session。
  2. 特徵工程（與 /predict、/mediapipe/predict 共用 session_features）：
     - 過濾 visibility < 0.5
     - 只保留跑走核心關節（去掉臉部／手指）
     - 每幀以 hip_mid 歸零並除以軀幹長度標準化
     - 關節座標取 mean／std；生物力學角度取 mean／std／min／max
  3. 以與推論相同的 window_size／stride 切時間窗，擴增樣本並對齊分布。
  4. GroupShuffleSplit（依 session id）避免同影片洩漏到測試集。
  5. 僅對訓練集做水平翻轉增強；GradientBoosting（balanced sample_weight）後印出訓練／測試報告、bad recall 與正確樣本平均信心度。

使用：
  python train.py                         # → pose_quality_model.joblib
  python train.py --engine mediapipe      # → mediapipe_quality_model.joblib
  python train.py --engine mediapipe --from-json mediapipe_export.json
  python train.py --engine mediapipe --from-api https://... --username 帳號
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

import joblib
import numpy as np
from mongo_util import describe_mongo_url_problem, make_mongo_client, mongo_url_help
from pymongo.errors import ConfigurationError
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

MONGO_URL = os.environ.get("POSE_MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("POSE_MONGO_DB", "pose")
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, "pose_quality_model.joblib")
MEDIAPIPE_MODEL_PATH = os.path.join(BACKEND_DIR, "mediapipe_quality_model.joblib")
CSV_PATH = os.path.join(BACKEND_DIR, "dataset.csv")
MEDIAPIPE_CSV_PATH = os.path.join(BACKEND_DIR, "mediapipe_dataset.csv")

# 與後端 _predict_with_engine 預設一致，訓練／推論必須同值。
DEFAULT_WINDOW_SIZE = 30
DEFAULT_WINDOW_STRIDE = 15
MIN_VISIBILITY = 0.5
MIN_TORSO_LEN = 1e-3
FEATURE_VERSION = 2

# 跑走核心關節（移除臉部 eye_*／mouth／ear 與手指 pinky／index／thumb）。
SIDES = ("left", "right")
PER_SIDE = (
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
    "heel",
    "foot_index",
)
JOINTS: list[str] = ["shoulder_mid", "hip_mid"] + [f"{s}_{p}" for s in SIDES for p in PER_SIDE]

ANGLE_NAMES: list[str] = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "torso_lean",
    "pelvis_tilt",
]

COORD_FEATURE_COLUMNS = [
    f"{j}_{stat}_{axis}" for j in JOINTS for stat in ("mean", "std") for axis in ("x", "y")
]
ANGLE_FEATURE_COLUMNS = [
    f"{a}_{stat}" for a in ANGLE_NAMES for stat in ("mean", "std", "min", "max")
]
FEATURE_COLUMNS = COORD_FEATURE_COLUMNS + ANGLE_FEATURE_COLUMNS


def frame_windows(
    frames: list,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> list[list]:
    """把影格切成重疊時間窗；影格不足一窗時回傳整段。與 main._predict 共用。"""
    n = len(frames)
    if n == 0:
        return []
    w = max(1, int(window_size))
    s = max(1, int(stride))
    if n <= w:
        return [frames]
    windows = [frames[i : i + w] for i in range(0, n - w + 1, s)]
    if windows[-1][-1] is not frames[-1]:
        windows.append(frames[-w:])
    return windows


def _node_visibility(node: dict) -> float:
    raw = node.get("visibility")
    if raw is None:
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def _as_xy(node: dict | None) -> np.ndarray | None:
    if node is None:
        return None
    try:
        return np.array([float(node["x"]), float(node["y"])], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """關節角 ∠ABC（度）。"""
    ba = a - b
    bc = c - b
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < 1e-8 or nc < 1e-8:
        return None
    cos = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _torso_lean_degrees(hip_mid: np.ndarray, shoulder_mid: np.ndarray) -> float:
    """軀幹相對垂直線的前傾／側傾角（影像 y 向下）。"""
    dx = float(shoulder_mid[0] - hip_mid[0])
    dy = float(shoulder_mid[1] - hip_mid[1])
    return math.degrees(math.atan2(dx, -dy))


def _pelvis_tilt_degrees(left_hip: np.ndarray, right_hip: np.ndarray) -> float:
    """骨盆水平傾斜角（左右髖連線相對水平）。"""
    dx = float(right_hip[0] - left_hip[0])
    dy = float(right_hip[1] - left_hip[1])
    return math.degrees(math.atan2(dy, dx))


def _swap_side_joint(joint: str) -> str:
    if joint.startswith("left_"):
        return "right_" + joint[5:]
    if joint.startswith("right_"):
        return "left_" + joint[6:]
    return joint


def horizontally_flip_frames(frames: list[dict]) -> list[dict]:
    """影像水平翻轉：x → 1-x，並對調 left_*／right_* 關節名。"""
    out: list[dict] = []
    for frame in frames:
        nodes = []
        for node in frame.get("nodes", []):
            if not isinstance(node, dict):
                continue
            flipped = dict(node)
            joint = str(node.get("joint") or "")
            flipped["joint"] = _swap_side_joint(joint)
            try:
                flipped["x"] = 1.0 - float(node["x"])
            except (KeyError, TypeError, ValueError):
                pass
            nodes.append(flipped)
        new_frame = dict(frame)
        new_frame["nodes"] = nodes
        out.append(new_frame)
    return out


def session_features(doc: dict) -> list[float] | None:
    """
    把一個 session／時間窗的多幀節點壓成固定長度特徵向量。
    訓練與 /predict、/mediapipe/predict 必須共用此函式。
    """
    frames = doc.get("frames", [])
    if not frames:
        return None

    coord_series: dict[str, list[tuple[float, float]]] = {j: [] for j in JOINTS}
    angle_series: dict[str, list[float]] = {a: [] for a in ANGLE_NAMES}

    for frame in frames:
        visible: dict[str, dict] = {}
        for node in frame.get("nodes", []):
            if not isinstance(node, dict):
                continue
            joint = node.get("joint")
            if joint not in coord_series:
                continue
            if _node_visibility(node) < MIN_VISIBILITY:
                continue
            visible[str(joint)] = node

        hip_mid = _as_xy(visible.get("hip_mid"))
        shoulder_mid = _as_xy(visible.get("shoulder_mid"))
        if hip_mid is None or shoulder_mid is None:
            continue

        torso = float(np.linalg.norm(shoulder_mid - hip_mid))
        scale = max(torso, MIN_TORSO_LEN)

        for joint in JOINTS:
            xy = _as_xy(visible.get(joint))
            if xy is None:
                continue
            norm = (xy - hip_mid) / scale
            coord_series[joint].append((float(norm[0]), float(norm[1])))

        left_hip = _as_xy(visible.get("left_hip"))
        right_hip = _as_xy(visible.get("right_hip"))
        left_knee = _as_xy(visible.get("left_knee"))
        right_knee = _as_xy(visible.get("right_knee"))
        left_ankle = _as_xy(visible.get("left_ankle"))
        right_ankle = _as_xy(visible.get("right_ankle"))
        left_shoulder = _as_xy(visible.get("left_shoulder"))
        right_shoulder = _as_xy(visible.get("right_shoulder"))

        if left_hip is not None and left_knee is not None and left_ankle is not None:
            ang = _angle_degrees(left_hip, left_knee, left_ankle)
            if ang is not None:
                angle_series["left_knee"].append(ang)
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            ang = _angle_degrees(right_hip, right_knee, right_ankle)
            if ang is not None:
                angle_series["right_knee"].append(ang)
        if left_shoulder is not None and left_hip is not None and left_knee is not None:
            ang = _angle_degrees(left_shoulder, left_hip, left_knee)
            if ang is not None:
                angle_series["left_hip"].append(ang)
        if right_shoulder is not None and right_hip is not None and right_knee is not None:
            ang = _angle_degrees(right_shoulder, right_hip, right_knee)
            if ang is not None:
                angle_series["right_hip"].append(ang)

        angle_series["torso_lean"].append(_torso_lean_degrees(hip_mid, shoulder_mid))
        if left_hip is not None and right_hip is not None:
            angle_series["pelvis_tilt"].append(_pelvis_tilt_degrees(left_hip, right_hip))

    if not any(coord_series[j] for j in JOINTS):
        return None

    feats: list[float] = []
    for joint in JOINTS:
        arr = np.array(coord_series[joint], dtype=float) if coord_series[joint] else np.zeros((1, 2))
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        feats.extend([float(mean[0]), float(std[0]), float(mean[1]), float(std[1])])

    for name in ANGLE_NAMES:
        vals = angle_series[name]
        if vals:
            arr = np.array(vals, dtype=float)
            feats.extend(
                [
                    float(arr.mean()),
                    float(arr.std()),
                    float(arr.min()),
                    float(arr.max()),
                ]
            )
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    if len(feats) != len(FEATURE_COLUMNS):
        raise RuntimeError(
            f"特徵維度不一致：got {len(feats)}, expected {len(FEATURE_COLUMNS)}"
        )
    return feats


def is_mediapipe_training_doc(doc: dict) -> bool:
    """判斷一筆已上傳 session 是否來自 MediaPipe（含回退寫入 pose_sessions 的資料）。"""
    engine = str(doc.get("engine") or "").strip().lower()
    if engine == "mediapipe":
        return True
    source = str(doc.get("source_label") or "").strip().lower()
    return source.startswith("mediapipe")


def labeled_feature_row(doc: dict) -> tuple[list[float], int] | None:
    label = doc.get("label")
    if label not in ("good", "bad"):
        return None
    feats = session_features(doc)
    if feats is None:
        return None
    return feats, (1 if label == "good" else 0)


def collect_training_docs(db, engine: str) -> list[dict]:
    """依引擎從 Mongo 收集訓練用 session。"""
    pose_col = db["pose_sessions"]
    mp_col = db["mediapipe_sessions"]
    docs: list[dict] = []
    if engine in ("mediapipe", "all"):
        docs.extend(mp_col.find({}))
        docs.extend(d for d in pose_col.find({}) if is_mediapipe_training_doc(d))
    if engine in ("pose", "all"):
        docs.extend(d for d in pose_col.find({}) if not is_mediapipe_training_doc(d))
    return docs


def rows_from_docs(
    docs: list[dict],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    將每筆 session 依與推論相同的時間窗切成多個樣本。
    回傳 (X, y, groups)，groups 為 session id，供 GroupShuffleSplit 使用。
    """
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    seen: set[str] = set()

    for index, doc in enumerate(docs):
        key = str(doc.get("_id") or doc.get("id") or f"session-{index}")
        if key in seen:
            continue
        seen.add(key)

        label = doc.get("label")
        if label not in ("good", "bad"):
            continue
        y_val = 1 if label == "good" else 0
        frames = doc.get("frames") or []
        if not frames:
            continue

        for window in frame_windows(frames, window_size=window_size, stride=stride):
            feats = session_features({"frames": window})
            if feats is None:
                continue
            features.append(feats)
            labels.append(y_val)
            groups.append(key)

    if not features:
        return (
            np.zeros((0, len(FEATURE_COLUMNS))),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=object),
        )
    return (
        np.array(features, dtype=float),
        np.array(labels, dtype=int),
        np.array(groups, dtype=object),
    )


def augment_train_with_horizontal_flip(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_docs_windows: list[tuple[list[dict], int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    僅對訓練集做水平翻轉增強。
    若提供原始時間窗影格，會翻轉後重算 session_features；否則回傳原資料。
    """
    if not train_docs_windows:
        return X_train, y_train
    extra_x: list[list[float]] = []
    extra_y: list[int] = []
    for frames, label in train_docs_windows:
        feats = session_features({"frames": horizontally_flip_frames(frames)})
        if feats is None:
            continue
        extra_x.append(feats)
        extra_y.append(label)
    if not extra_x:
        return X_train, y_train
    return (
        np.vstack([X_train, np.array(extra_x, dtype=float)]),
        np.concatenate([y_train, np.array(extra_y, dtype=int)]),
    )


def load_dataset(
    engine: str = "pose",
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    problem = describe_mongo_url_problem(MONGO_URL)
    if problem:
        raise RuntimeError(mongo_url_help(MONGO_URL))
    try:
        client = make_mongo_client(MONGO_URL)
    except ConfigurationError as exc:
        raise RuntimeError(mongo_url_help(MONGO_URL, exc)) from exc
    try:
        db = client[MONGO_DB_NAME]
        return rows_from_docs(
            collect_training_docs(db, engine),
            window_size=window_size,
            stride=stride,
        )
    finally:
        client.close()


def load_docs_from_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        sessions = payload.get("sessions")
        if isinstance(sessions, list):
            return sessions
    raise RuntimeError("JSON 格式不正確：需要 {\"sessions\": [...]} 或 session 陣列")


def load_dataset_from_json(
    path: str,
    engine: str,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    docs = load_docs_from_json(path)
    if engine == "mediapipe":
        docs = [doc for doc in docs if is_mediapipe_training_doc(doc) or str(doc.get("engine") or "") == "mediapipe"]
        if not docs:
            docs = load_docs_from_json(path)
    elif engine == "pose":
        docs = [doc for doc in docs if not is_mediapipe_training_doc(doc)]
    print(f"從 JSON 讀取：{path}（{len(docs)} 筆 session）")
    return rows_from_docs(docs, window_size=window_size, stride=stride)


def _http_json(method: str, url: str, *, body: dict | None = None, headers: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {url}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"無法連線 {url}：{exc.reason}") from exc
    return json.loads(raw) if raw else {}


def fetch_docs_from_api(base_url: str, username: str, password: str, engine: str) -> list[dict]:
    base = base_url.rstrip("/")
    login = _http_json(
        "POST",
        f"{base}/auth/login",
        body={"username": username, "password": password},
    )
    token = str(login.get("access_token") or "").strip()
    if not token:
        raise RuntimeError("登入成功但沒有 access_token")
    payload = _http_json(
        "GET",
        f"{base}/dataset/export?engine={engine}",
        headers={"Authorization": f"Bearer {token}"},
    )
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("雲端 /dataset/export 回傳格式不正確（可能尚未部署此 API）")
    print(
        f"雲端匯出：{payload.get('count', len(sessions))} 筆"
        f"（good={payload.get('by_label', {}).get('good', '?')},"
        f" bad={payload.get('by_label', {}).get('bad', '?')}）"
    )
    return sessions


def load_dataset_from_api(
    base_url: str,
    username: str,
    password: str,
    engine: str,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rows_from_docs(
        fetch_docs_from_api(base_url, username, password, engine),
        window_size=window_size,
        stride=stride,
    )


def empty_dataset_hint(engine: str, mongo_url: str, from_api: bool) -> str:
    lines = [
        "樣本太少（至少需要約 4 個時間窗，且兩種標籤都要有；建議至少 2 筆以上不同 session）。",
        "",
        "手機 App「標記為好／壞」上傳的是 Railway 後面的 MongoDB Atlas，",
        "不是這台電腦預設的 mongodb://localhost:27017。",
        "本機 Realm 標籤也不會自動同步到雲端以外的資料庫。",
    ]
    if from_api:
        lines += [
            "",
            "若 --from-api 仍是 0 筆：請確認 App 詳情頁有出現「上傳成功」，",
            "且登入的是同一個帳號；雲端需已部署含 /dataset/export 的後端。",
            "也可以在 App 資料庫按「匯出訓練 JSON」，再：",
            f"  python train.py --engine {engine} --from-json 檔案.json",
        ]
    elif "localhost" in mongo_url or "127.0.0.1" in mongo_url:
        lines += [
            "",
            "你現在連的是本機空庫。不要複製文件裡的 mongodb+srv://...",
            "請到 Railway → Variables 複製完整 POSE_MONGO_URL，或從手機匯出 JSON：",
            f"  python train.py --engine {engine} --from-json 檔案.json",
        ]
    else:
        lines += [
            "",
            "請到 App 節點庫點進該筆，按「標記為好 (good)／壞 (bad)」，",
            "畫面上要出現「上傳成功」才會進雲端。",
        ]
    return "\n".join(lines)


def export_csv(X: np.ndarray, y: np.ndarray, path: str) -> None:
    header = ",".join(FEATURE_COLUMNS + ["label"])
    rows = [header]
    for feats, label in zip(X, y):
        rows.append(",".join(f"{v:.6f}" for v in feats) + f",{'good' if label == 1 else 'bad'}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[匯出] 已寫出資料集：{path}（{len(X)} 筆，每筆 {len(FEATURE_COLUMNS)} 維特徵）")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練姿勢品質模型")
    parser.add_argument(
        "--engine",
        choices=("pose", "mediapipe", "all"),
        default="pose",
        help="pose=QuickPose／自訓模型；mediapipe=原生 MediaPipe 引擎蒐集資料；all=合併",
    )
    parser.add_argument("--export", action="store_true", help="另外匯出 CSV")
    parser.add_argument(
        "--from-api",
        metavar="URL",
        help="改從雲端 GET /dataset/export 拉目前帳號已上傳的標籤資料（不連本機 Mongo）",
    )
    parser.add_argument("--username", help="--from-api 登入帳號（也可用環境變數 POSE_API_USERNAME）")
    parser.add_argument("--password", help="--from-api 登入密碼（也可用環境變數 POSE_API_PASSWORD）")
    parser.add_argument("--from-json", metavar="FILE", help="改從 App 匯出的訓練 JSON 讀資料（不連 Mongo）")
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"時間窗長度（預設 {DEFAULT_WINDOW_SIZE}，需與 /predict 一致）",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=DEFAULT_WINDOW_STRIDE,
        help=f"時間窗步長（預設 {DEFAULT_WINDOW_STRIDE}，需與 /predict 一致）",
    )
    return parser.parse_args(argv)


def model_output_path(engine: str) -> str:
    return MEDIAPIPE_MODEL_PATH if engine == "mediapipe" else MODEL_PATH


def csv_output_path(engine: str) -> str:
    return MEDIAPIPE_CSV_PATH if engine == "mediapipe" else CSV_PATH


def _mean_confidence_on_correct(model, X_test: np.ndarray, y_test: np.ndarray) -> float | None:
    """測試集中「預測正確」樣本的平均 predict_proba 信心（該預測類別的機率）。"""
    if len(X_test) == 0:
        return None
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    confs: list[float] = []
    for i, label in enumerate(pred):
        if label != y_test[i]:
            continue
        idx = classes.index(label)
        confs.append(float(proba[i, idx]))
    if not confs:
        return None
    return float(np.mean(confs))


def _collect_window_frames_by_group(
    docs: list[dict],
    groups_train: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> list[tuple[list[dict], int]]:
    """取出落在訓練組的時間窗原始影格，供水平翻轉增強。"""
    train_group_set = set(str(g) for g in groups_train.tolist())
    windows: list[tuple[list[dict], int]] = []
    seen: set[str] = set()
    for index, doc in enumerate(docs):
        key = str(doc.get("_id") or doc.get("id") or f"session-{index}")
        if key in seen or key not in train_group_set:
            continue
        seen.add(key)
        label = doc.get("label")
        if label not in ("good", "bad"):
            continue
        y_val = 1 if label == "good" else 0
        for window in frame_windows(doc.get("frames") or [], window_size=window_size, stride=stride):
            if session_features({"frames": window}) is None:
                continue
            windows.append((window, y_val))
    return windows


def _load_raw_docs_for_augment(args: argparse.Namespace, engine: str) -> list[dict]:
    if args.from_json:
        docs = load_docs_from_json(args.from_json)
        if engine == "mediapipe":
            filtered = [
                doc
                for doc in docs
                if is_mediapipe_training_doc(doc) or str(doc.get("engine") or "") == "mediapipe"
            ]
            return filtered or docs
        if engine == "pose":
            return [doc for doc in docs if not is_mediapipe_training_doc(doc)]
        return docs
    if args.from_api:
        username = (args.username or os.environ.get("POSE_API_USERNAME") or "").strip()
        password = args.password or os.environ.get("POSE_API_PASSWORD") or ""
        return fetch_docs_from_api(args.from_api, username, password, engine)
    problem = describe_mongo_url_problem(MONGO_URL)
    if problem:
        raise RuntimeError(mongo_url_help(MONGO_URL))
    client = make_mongo_client(MONGO_URL)
    try:
        return list(collect_training_docs(client[MONGO_DB_NAME], engine))
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = args.engine
    out_model = model_output_path(engine)
    out_csv = csv_output_path(engine)
    window_size = max(1, int(args.window_size))
    stride = max(1, int(args.window_stride))

    if args.from_json:
        print(f"從 JSON 訓練：{args.from_json} / engine={engine}")
        try:
            X, y, groups = load_dataset_from_json(
                args.from_json, engine, window_size=window_size, stride=stride
            )
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"無法讀取 JSON：{exc}")
            return 1
    elif args.from_api:
        username = (args.username or os.environ.get("POSE_API_USERNAME") or "").strip()
        password = args.password or os.environ.get("POSE_API_PASSWORD") or ""
        if not username or not password:
            print("使用 --from-api 時請提供 --username / --password，或設 POSE_API_USERNAME、POSE_API_PASSWORD。")
            return 1
        print(f"從雲端 API 拉資料：{args.from_api} / engine={engine} / user={username}")
        try:
            X, y, groups = load_dataset_from_api(
                args.from_api, username, password, engine, window_size=window_size, stride=stride
            )
        except RuntimeError as exc:
            print(f"無法從雲端匯出訓練資料：{exc}")
            print(empty_dataset_hint(engine, MONGO_URL, from_api=True))
            return 1
    else:
        print(f"連線 MongoDB：{MONGO_URL} / db={MONGO_DB_NAME} / engine={engine}")
        try:
            X, y, groups = load_dataset(engine, window_size=window_size, stride=stride)
        except RuntimeError as exc:
            print(exc)
            return 1

    n_sessions = len(set(groups.tolist())) if len(groups) else 0
    print(
        f"載入時間窗樣本：{len(X)}（session={n_sessions}, "
        f"good={int((y == 1).sum()) if len(y) else 0}, "
        f"bad={int((y == 0).sum()) if len(y) else 0}, "
        f"特徵維度={len(FEATURE_COLUMNS)}, "
        f"window={window_size}/{stride}）"
    )

    if args.export and len(X) > 0:
        export_csv(X, y, out_csv)

    if len(X) < 4:
        print(empty_dataset_hint(engine, MONGO_URL, from_api=bool(args.from_api)))
        return 1
    if len(set(y.tolist())) < 2:
        print("目前只有單一標籤，無法做二元分類。請補上另一種標籤的資料。")
        return 1
    if n_sessions < 2:
        print("至少需要 2 筆不同 session，才能做 GroupShuffleSplit（避免同影片洩漏）。")
        return 1

    test_size = 0.25
    if n_sessions < 4:
        test_size = 1.0 / n_sessions
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    try:
        raw_docs = _load_raw_docs_for_augment(args, engine)
        train_windows = _collect_window_frames_by_group(
            raw_docs, groups_train, window_size=window_size, stride=stride
        )
        before = len(X_train)
        X_train, y_train = augment_train_with_horizontal_flip(
            X_train, y_train, train_windows
        )
        print(f"水平翻轉增強：訓練集 {before} → {len(X_train)}")
    except Exception as exc:  # noqa: BLE001 — 增強失敗不阻斷訓練
        print(f"水平翻轉增強略過：{exc}")

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    classifier_name = "GradientBoostingClassifier(sample_weight=balanced)"
    sample_weight = compute_sample_weight("balanced", y_train)
    print(f"訓練模型：{classifier_name}")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, pred_test)
    bad_recall = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    print(f"\n訓練集準確率：{accuracy_score(y_train, pred_train):.3f}")
    print("訓練集分類報告：")
    print(classification_report(y_train, pred_train, target_names=["bad", "good"], zero_division=0))
    print(f"測試集準確率：{test_acc:.3f}")
    print(f"測試集 bad recall：{bad_recall:.3f}")
    print("測試集分類報告：")
    print(classification_report(y_test, pred_test, target_names=["bad", "good"], zero_division=0))

    mean_correct_conf = _mean_confidence_on_correct(model, X_test, y_test)
    if mean_correct_conf is None:
        print("測試集預測正確樣本的平均信心度：無（沒有預測正確的樣本）")
    else:
        print(
            f"測試集預測正確樣本的平均信心度：{mean_correct_conf:.3f}"
            f"（目標 ≥ 0.85；{'達標' if mean_correct_conf >= 0.85 else '未達標，可再補資料或調參'}）"
        )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "joints": JOINTS,
            "angle_names": ANGLE_NAMES,
            "engine": engine,
            "feature_version": FEATURE_VERSION,
            "window_size": window_size,
            "window_stride": stride,
            "n_features": len(FEATURE_COLUMNS),
            "classifier": classifier_name,
            "mean_correct_confidence": mean_correct_conf,
            "test_accuracy": float(test_acc),
            "test_bad_recall": float(bad_recall),
        },
        out_model,
    )
    print(f"[存檔] 模型已儲存：{out_model}（{len(FEATURE_COLUMNS)} 維，feature_version={FEATURE_VERSION}）")
    if engine == "mediapipe":
        print("部署後請重啟後端，App 的 MediaPipe／自訓 MediaPipe 模型會呼叫 /mediapipe/predict。")
    else:
        print("部署後請重啟後端，讓 /predict 載入新特徵維度的模型。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

