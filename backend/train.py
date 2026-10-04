"""
姿勢品質（好 / 壞）監督式學習訓練腳本。

流程：
  1. 從 MongoDB 讀出已標註的姿勢 session。
  2. 特徵工程：每個 session 把每個關節在所有影格的 (x, y, z) 取「平均與標準差」，
     組成固定長度的特徵向量（35 關節 × 6 = 210 維）。
  3. 以 RandomForest 訓練二元分類器（good=1 / bad=0），輸出準確率與報告。
  4. 將模型存成 joblib（pose 或 mediapipe 分開）。

使用：
  python train.py                         # 訓練 QuickPose／自訓模型資料 → pose_quality_model.joblib
  python train.py --engine mediapipe      # 訓練 MediaPipe 蒐集資料 → mediapipe_quality_model.joblib
  python train.py --engine mediapipe --export
  python train.py --engine all            # 兩套資料合併成 pose_quality_model.joblib

  # 手機上傳到 Railway／Atlas，本機 train.py 預設連 localhost（通常是空的）。
  # 請用和 Railway 相同的 Atlas 連線，或從雲端 API 拉你帳號標好的資料：
  # 從 Railway Variables 複製完整 POSE_MONGO_URL，不要用 mongodb+srv://...
  export POSE_MONGO_URL="mongodb+srv://使用者:密碼@cluster0.xxxx.mongodb.net/"
  python train.py --engine mediapipe
  python train.py --engine mediapipe --from-json mediapipe_export.json
  python train.py --engine mediapipe --from-api https://runpose-backend-production.up.railway.app --username 帳號
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import joblib
import numpy as np
from mongo_util import describe_mongo_url_problem, make_mongo_client, mongo_url_help
from pymongo.errors import ConfigurationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MONGO_URL = os.environ.get("POSE_MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("POSE_MONGO_DB", "pose")
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, "pose_quality_model.joblib")
MEDIAPIPE_MODEL_PATH = os.path.join(BACKEND_DIR, "mediapipe_quality_model.joblib")
CSV_PATH = os.path.join(BACKEND_DIR, "dataset.csv")
MEDIAPIPE_CSV_PATH = os.path.join(BACKEND_DIR, "mediapipe_dataset.csv")

# 關節順序需與 iOS 端 PoseNodeExtractor 一致（共 35 個）。
SIDES = ["left", "right"]
PER_SIDE = [
    "eye_inner", "eye", "eye_outer", "ear", "mouth",
    "shoulder", "elbow", "wrist", "pinky", "index", "thumb",
    "hip", "knee", "ankle", "heel", "foot_index",
]
JOINTS = ["nose", "shoulder_mid", "hip_mid"] + [f"{s}_{p}" for s in SIDES for p in PER_SIDE]

# 每個關節的特徵欄位：x/y/z 的平均與標準差。
FEATURE_COLUMNS = [f"{j}_{stat}_{axis}" for j in JOINTS for stat in ("mean", "std") for axis in ("x", "y", "z")]


def session_features(doc: dict) -> list[float] | None:
    """把一個 session 的多幀節點壓成固定長度特徵向量。"""
    frames = doc.get("frames", [])
    if not frames:
        return None

    acc: dict[str, list[tuple[float, float, float]]] = {j: [] for j in JOINTS}
    for frame in frames:
        for node in frame.get("nodes", []):
            joint = node.get("joint")
            if joint in acc:
                acc[joint].append((node["x"], node["y"], node["z"]))

    feats: list[float] = []
    for joint in JOINTS:
        arr = np.array(acc[joint], dtype=float) if acc[joint] else np.zeros((1, 3))
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        feats.extend([mean[0], std[0], mean[1], std[1], mean[2], std[2]])
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


def rows_from_docs(docs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    seen: set[str] = set()
    for doc in docs:
        key = str(doc.get("_id", ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        row = labeled_feature_row(doc)
        if row is None:
            continue
        feats, label = row
        features.append(feats)
        labels.append(label)
    if not features:
        return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0,), dtype=int)
    return np.array(features, dtype=float), np.array(labels, dtype=int)


def load_dataset(engine: str = "pose") -> tuple[np.ndarray, np.ndarray]:
    problem = describe_mongo_url_problem(MONGO_URL)
    if problem:
        raise RuntimeError(mongo_url_help(MONGO_URL))
    try:
        client = make_mongo_client(MONGO_URL)
    except ConfigurationError as exc:
        raise RuntimeError(mongo_url_help(MONGO_URL, exc)) from exc
    try:
        db = client[MONGO_DB_NAME]
        return rows_from_docs(collect_training_docs(db, engine))
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


def load_dataset_from_json(path: str, engine: str) -> tuple[np.ndarray, np.ndarray]:
    docs = load_docs_from_json(path)
    if engine == "mediapipe":
        docs = [doc for doc in docs if is_mediapipe_training_doc(doc) or str(doc.get("engine") or "") == "mediapipe"]
        if not docs:
            docs = load_docs_from_json(path)
    elif engine == "pose":
        docs = [doc for doc in docs if not is_mediapipe_training_doc(doc)]
    print(f"從 JSON 讀取：{path}（{len(docs)} 筆 session）")
    return rows_from_docs(docs)


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


def load_dataset_from_api(base_url: str, username: str, password: str, engine: str) -> tuple[np.ndarray, np.ndarray]:
    return rows_from_docs(fetch_docs_from_api(base_url, username, password, engine))


def empty_dataset_hint(engine: str, mongo_url: str, from_api: bool) -> str:
    lines = [
        "樣本太少（至少需要約 4 筆，且兩種標籤都要有）。",
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
        help="pose=QuickPose／自訓模型；mediapipe=MediaPipe 蒐集資料；all=合併",
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
    return parser.parse_args(argv)


def model_output_path(engine: str) -> str:
    return MEDIAPIPE_MODEL_PATH if engine == "mediapipe" else MODEL_PATH


def csv_output_path(engine: str) -> str:
    return MEDIAPIPE_CSV_PATH if engine == "mediapipe" else CSV_PATH


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = args.engine
    out_model = model_output_path(engine)
    out_csv = csv_output_path(engine)

    if args.from_json:
        print(f"從 JSON 訓練：{args.from_json} / engine={engine}")
        try:
            X, y = load_dataset_from_json(args.from_json, engine)
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
            X, y = load_dataset_from_api(args.from_api, username, password, engine)
        except RuntimeError as exc:
            print(f"無法從雲端匯出訓練資料：{exc}")
            print(empty_dataset_hint(engine, MONGO_URL, from_api=True))
            return 1
    else:
        print(f"連線 MongoDB：{MONGO_URL} / db={MONGO_DB_NAME} / engine={engine}")
        try:
            X, y = load_dataset(engine)
        except RuntimeError as exc:
            print(exc)
            return 1
    print(f"載入樣本數：{len(X)}（good={int((y == 1).sum()) if len(y) else 0}, bad={int((y == 0).sum()) if len(y) else 0}）")

    if args.export and len(X) > 0:
        export_csv(X, y, out_csv)

    if len(X) < 4:
        print(empty_dataset_hint(engine, MONGO_URL, from_api=bool(args.from_api)))
        return 1
    if len(set(y.tolist())) < 2:
        print("目前只有單一標籤，無法做二元分類。請補上另一種標籤的資料。")
        return 1

    stratify = y if min(int((y == 1).sum()), int((y == 0).sum())) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"\n測試集準確率：{accuracy_score(y_test, pred):.3f}")
    print("\n分類報告：")
    print(classification_report(y_test, pred, target_names=["bad", "good"], zero_division=0))

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "joints": JOINTS,
            "engine": engine,
        },
        out_model,
    )
    print(f"[存檔] 模型已儲存：{out_model}")
    if engine == "mediapipe":
        print("部署後請重啟後端，App 的 MediaPipe 引擎會呼叫 /mediapipe/predict。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
