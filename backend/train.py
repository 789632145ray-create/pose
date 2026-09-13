"""
姿勢品質（好 / 壞）監督式學習訓練腳本。

流程：
  1. 從 MongoDB 讀出所有已標註的姿勢 session。
  2. 特徵工程：每個 session 把每個關節在所有影格的 (x, y, z) 取「平均與標準差」，
     組成固定長度的特徵向量（35 關節 × 6 = 210 維）。
  3. 以 RandomForest 訓練二元分類器（good=1 / bad=0），輸出準確率與報告。
  4. 將模型存成 pose_quality_model.joblib，並可一併匯出資料集 CSV。

使用：
  python train.py            # 訓練 + 評估 + 存模型
  python train.py --export   # 另外匯出 dataset.csv（特徵 + 標籤）
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
from mongo_util import make_mongo_client
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MONGO_URL = os.environ.get("POSE_MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("POSE_MONGO_DB", "pose")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_quality_model.joblib")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

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


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    client = make_mongo_client(MONGO_URL)
    collection = client[MONGO_DB_NAME]["pose_sessions"]

    features: list[list[float]] = []
    labels: list[int] = []
    for doc in collection.find({}):
        label = doc.get("label")
        if label not in ("good", "bad"):
            continue
        feats = session_features(doc)
        if feats is None:
            continue
        features.append(feats)
        labels.append(1 if label == "good" else 0)

    return np.array(features, dtype=float), np.array(labels, dtype=int)


def export_csv(X: np.ndarray, y: np.ndarray) -> None:
    header = ",".join(FEATURE_COLUMNS + ["label"])
    rows = [header]
    for feats, label in zip(X, y):
        rows.append(",".join(f"{v:.6f}" for v in feats) + f",{'good' if label == 1 else 'bad'}")
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"[匯出] 已寫出資料集：{CSV_PATH}（{len(X)} 筆，每筆 {len(FEATURE_COLUMNS)} 維特徵）")


def main() -> None:
    do_export = "--export" in sys.argv

    print(f"連線 MongoDB：{MONGO_URL} / db={MONGO_DB_NAME}")
    X, y = load_dataset()
    print(f"載入樣本數：{len(X)}（good={int((y == 1).sum())}, bad={int((y == 0).sum())}）")

    if do_export and len(X) > 0:
        export_csv(X, y)

    if len(X) < 4:
        print("樣本太少（至少需要約 4 筆，且兩種標籤都要有）。請先用 App 上傳更多『好/壞』資料再訓練。")
        return
    if len(set(y.tolist())) < 2:
        print("目前只有單一標籤，無法做二元分類。請補上另一種標籤的資料。")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"\n測試集準確率：{accuracy_score(y_test, pred):.3f}")
    print("\n分類報告：")
    print(classification_report(y_test, pred, target_names=["bad", "good"], zero_division=0))

    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "joints": JOINTS}, MODEL_PATH)
    print(f"[存檔] 模型已儲存：{MODEL_PATH}")


if __name__ == "__main__":
    main()
