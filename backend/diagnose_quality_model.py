"""
姿勢品質模型診斷：比對 train／inference 前處理、標籤分佈、小樣本過擬合。

用途（對應「訓練資料上信心分數仍很低」）：
  1. 確認 Normalization／縮放／特徵欄位順序在 train 與 /predict 是否 100% 一致。
  2. 印出訓練集 Labels（good／bad）數量比例，檢查資料不平衡。
  3. 只取 10 筆訓練資料，用 MLP 過度擬合 100 個 Epochs，目標訓練集信心 ≥ 0.99，
     以排除特徵管線／標籤編碼／Loss（交叉熵）明顯寫錯。

注意：正式線上模型是 RandomForest（無 Epoch）。本腳本的 100 Epochs 是刻意的過擬合
檢查；同一批 10 筆也會跑 RF 記憶測試作為對照。

使用（不必先有手機匯出的 JSON；優先 --from-api）：
  cd backend && source .venv/bin/activate
  # App「節點資料庫」上傳走 /poses；MediaPipe 庫才是 mediapipe。不確定就用 all。
  python diagnose_quality_model.py --engine all \\
    --from-api https://runpose-backend-production.up.railway.app \\
    --username 你的帳號 --save-json training_from_api.json
  # 密碼：--password 或環境變數 POSE_API_PASSWORD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

from train import (
    FEATURE_COLUMNS,
    JOINTS,
    feature_layout_report,
    load_dataset,
    load_dataset_from_api,
    load_dataset_from_json,
    parse_args as parse_train_args,
    print_label_class_ratios,
    session_features,
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
OVERFIT_TARGET = 0.99
OVERFIT_EPOCHS = 100
OVERFIT_N = 10


def _synthetic_docs(n: int = 10) -> list[dict]:
    """產生可分離的假資料，讓無 Mongo／JSON 時仍能跑過擬合檢查。"""
    docs: list[dict] = []
    for i in range(n):
        label = "good" if i % 2 == 0 else "bad"
        base = 0.7 if label == "good" else 0.2
        frames = []
        for f in range(8):
            nodes = []
            for j_idx, joint in enumerate(JOINTS):
                # good／bad 在 nose／hip 有明顯位移，其餘關節固定。
                dx = 0.05 * (1 if label == "good" else -1) if joint in {"nose", "hip_mid", "left_hip", "right_hip"} else 0.0
                nodes.append(
                    {
                        "joint": joint,
                        "x": base + dx + 0.001 * j_idx + 0.0001 * f,
                        "y": 0.4 + 0.001 * (n - i) + 0.0002 * f,
                        "z": -0.1 + 0.0001 * f,
                        "visibility": 0.9,
                        "presence": 0.9,
                    }
                )
            frames.append({"frame_index": f, "timestamp": float(f) * 0.033, "nodes": nodes})
        docs.append({"_id": f"synth-{i}", "label": label, "frames": frames, "engine": "pose"})
    return docs


def _maybe_save_api_json(args: argparse.Namespace, sessions: list[dict]) -> None:
    path = getattr(args, "save_json", None)
    if not path:
        return
    payload = {
        "engine": args.engine,
        "count": len(sessions),
        "by_label": {
            "good": sum(1 for doc in sessions if doc.get("label") == "good"),
            "bad": sum(1 for doc in sessions if doc.get("label") == "bad"),
        },
        "sessions": sessions,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[資料] 已把雲端匯出寫成 JSON：{path}（之後可用 --from-json 重跑）")


def load_xy(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    """與 train.py 相同資料來源。

    注意：--from-api 若回傳 0 筆，預設直接失敗（不再默默改用合成資料），
    避免「其實雲端有資料、只是 engine／帳號錯了」被誤判成沒東西。
    """
    from train import fetch_docs_from_api, rows_from_docs

    engine = args.engine
    if args.from_json:
        X, y = load_dataset_from_json(args.from_json, engine)
        source = f"json:{args.from_json}"
    elif args.from_api:
        username = (args.username or os.environ.get("POSE_API_USERNAME") or "").strip()
        password = args.password or os.environ.get("POSE_API_PASSWORD") or ""
        if not username or not password:
            raise RuntimeError(
                "使用 --from-api 時請提供 --username / --password（或 POSE_API_USERNAME、POSE_API_PASSWORD）。\n"
                "這不需要手機匯出 JSON：登入後會打雲端 GET /dataset/export。"
            )
        sessions = fetch_docs_from_api(args.from_api, username, password, engine)
        _maybe_save_api_json(args, sessions)
        X, y = rows_from_docs(sessions)
        source = f"api:{args.from_api}/engine={engine}/user={username}"
        if len(sessions) == 0:
            raise RuntimeError(
                f"雲端帳號「{username}」在 engine={engine} 下沒有已標註 session。\n"
                "截圖若是「節點資料庫／偵測詳情」且已上傳成功，資料在 /poses，請改：\n"
                f"  python diagnose_quality_model.py --engine all --from-api {args.from_api} "
                f"--username {username} --save-json training_from_api.json\n"
                "並確認登入的是同一個帳號（上傳成功那組）。"
            )
        if len(X) == 0 and sessions:
            raise RuntimeError(
                f"雲端有 {len(sessions)} 筆 session，但沒有可用的 good/bad 特徵列（可能 frames 為空）。"
            )
        return X, y, source
    else:
        try:
            X, y = load_dataset(engine)
            source = f"mongo:{os.environ.get('POSE_MONGO_URL', 'mongodb://localhost:27017')}"
        except RuntimeError as exc:
            print(f"[資料] Mongo 無法使用（{exc}）。")
            print(
                "若資料已上傳 Railway，請改用（不必 JSON）：\n"
                "  python diagnose_quality_model.py --engine all \\\n"
                "    --from-api https://runpose-backend-production.up.railway.app \\\n"
                "    --username 你的帳號 --save-json training_from_api.json"
            )
            if not args.allow_synthetic:
                raise RuntimeError("未加 --allow-synthetic，不改用合成資料。") from exc
            print("暫時改用合成 10 筆做過擬合檢查。")
            X, y = rows_from_docs(_synthetic_docs(OVERFIT_N))
            source = "synthetic"
            return X, y, source

    if len(X) == 0:
        print("[資料] 沒有已標註樣本。")
        if args.from_json:
            raise RuntimeError("JSON 裡沒有 good/bad 標註列。")
        if not args.allow_synthetic:
            raise RuntimeError(
                "沒有訓練列。請改用 --from-api --engine all，或加 --allow-synthetic 只測過擬合管線。"
            )
        print("暫時改用合成 10 筆做過擬合檢查。")
        X, y = rows_from_docs(_synthetic_docs(OVERFIT_N))
        source = "synthetic"
    return X, y, source


def check_train_infer_preprocess() -> dict[str, Any]:
    """比對 train 與 inference（main._predict_with_engine）的前處理是否一致。"""
    layout = feature_layout_report()
    print("=" * 60)
    print("1) Train vs Inference 前處理比對")
    print("=" * 60)
    print(f"  特徵維度：{layout['feature_dim']}（35 關節 × 6）")
    print(f"  Normalization：{layout['normalization']}")
    print(f"  Scaling：{layout['scaling']}")
    print(f"  使用 visibility／presence：{layout['uses_visibility']}")
    print(f"  缺關節填補：{layout['missing_joint_fill']}")
    print(f"  每關節欄位順序：{', '.join(layout['per_joint_layout'])}")
    print(f"  FEATURE_COLUMNS 前 6 欄：{FEATURE_COLUMNS[:6]}")

    # inference 必須 import 同一個 session_features，不可另寫一份（讀原始碼，避免啟動 FastAPI）。
    main_path = os.path.join(BACKEND_DIR, "main.py")
    with open(main_path, encoding="utf-8") as handle:
        main_src = handle.read()
    # 只檢查 _predict_with_engine 區段。
    marker = "def _predict_with_engine"
    start = main_src.find(marker)
    end = main_src.find("\n@app.post", start + 1)
    predict_src = main_src[start:end if end > start else start + 800]
    uses_shared = "from train import session_features" in predict_src and "session_features(" in predict_src
    no_local_scaler = "StandardScaler" not in predict_src and "MinMax" not in predict_src
    no_reorder = "feature_columns" not in predict_src  # 不依名稱重排，靠同一函式輸出順序

    doc = {
        "frames": [
            {
                "nodes": [
                    {"joint": "nose", "x": 0.51, "y": 0.22, "z": -0.05},
                    {"joint": "left_hip", "x": 0.40, "y": 0.61, "z": 0.02},
                ]
            }
        ]
    }
    feats = session_features(doc)
    assert feats is not None
    layout_ok = len(feats) == len(FEATURE_COLUMNS) == 210
    # nose：mean_x=0.51, std_x=0, mean_y=0.22, std_y=0, mean_z=-0.05, std_z=0
    nose_ok = (
        abs(feats[0] - 0.51) < 1e-9
        and abs(feats[1] - 0.0) < 1e-9
        and abs(feats[2] - 0.22) < 1e-9
        and abs(feats[3] - 0.0) < 1e-9
        and abs(feats[4] - (-0.05)) < 1e-9
        and abs(feats[5] - 0.0) < 1e-9
    )

    result = {
        "shared_session_features": uses_shared,
        "no_infer_only_scaler": no_local_scaler,
        "no_column_reorder_on_infer": no_reorder,
        "feature_dim_ok": layout_ok,
        "feature_column_order_matches_vector": nose_ok,
        "layout": layout,
    }
    all_ok = all(
        [
            uses_shared,
            no_local_scaler,
            no_reorder,
            layout_ok,
            nose_ok,
        ]
    )
    print(f"  /predict 使用 train.session_features：{'OK' if uses_shared else 'FAIL'}")
    print(f"  /predict 無額外 Scaler：{'OK' if no_local_scaler else 'FAIL'}")
    print(f"  /predict 不重排欄位：{'OK' if no_reorder else 'FAIL'}")
    print(f"  特徵維度 210 且與 FEATURE_COLUMNS 對齊：{'OK' if layout_ok else 'FAIL'}")
    print(f"  實際向量順序 = FEATURE_COLUMNS 名稱順序：{'OK' if nose_ok else 'FAIL'}")
    print(f"  → 前處理一致性：{'100% 一致' if all_ok else '不一致，請修正'}")
    result["ok"] = all_ok
    return result


def check_label_balance(y: np.ndarray) -> dict[str, Any]:
    print()
    print("=" * 60)
    print("2) 訓練集 Labels 數量比例")
    print("=" * 60)
    stats = print_label_class_ratios(y)
    return stats


def _mean_true_class_confidence(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    scores = []
    for row, label in zip(proba, y):
        idx = class_to_idx[int(label)]
        scores.append(float(row[idx]))
    return float(np.mean(scores)) if scores else 0.0


def overfit_sanity_check(X: np.ndarray, y: np.ndarray, n: int = OVERFIT_N, epochs: int = OVERFIT_EPOCHS) -> dict[str, Any]:
    """取最多 n 筆做 MLP 100 epoch 過擬合；目標平均 true-class 信心 ≥ 0.99。"""
    print()
    print("=" * 60)
    print(f"3) 過擬合檢查（最多 {n} 筆，{epochs} Epochs，目標信心 ≥ {OVERFIT_TARGET}）")
    print("=" * 60)

    if len(X) == 0:
        print("  無資料，跳過。")
        return {"ok": False, "reason": "empty"}

    # 盡量兩邊標籤都取到。
    good_idx = np.where(y == 1)[0]
    bad_idx = np.where(y == 0)[0]
    take_good = min(len(good_idx), n // 2 + n % 2)
    take_bad = min(len(bad_idx), n - take_good)
    if take_good + take_bad < n:
        # 單類時就直接從頭取 n 筆。
        idx = np.arange(min(n, len(X)))
    else:
        idx = np.concatenate([good_idx[:take_good], bad_idx[:take_bad]])
    idx = idx[:n]
    Xs, ys = X[idx], y[idx]
    print(f"  使用樣本：{len(Xs)}（good={int((ys == 1).sum())}, bad={int((ys == 0).sum())}）")

    # RF 對照：能背起來代表特徵／標籤沒有對調。
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(Xs, ys)
    rf_pred = rf.predict(Xs)
    rf_acc = float(accuracy_score(ys, rf_pred))
    rf_proba = rf.predict_proba(Xs)
    rf_conf = _mean_true_class_confidence(rf_proba, ys, rf.classes_)
    print(f"  RandomForest 訓練準確率：{rf_acc:.3f}（對照；RF 機率通常不會極尖）")
    print(f"  RandomForest 平均 true-class 信心：{rf_conf:.3f}")

    # MLP + log_loss（交叉熵）跑固定 Epochs。
    if len(set(ys.tolist())) < 2:
        print("  只有單一標籤，改以 RF 準確率判斷；MLP 二元過擬合跳過。")
        ok = rf_acc >= OVERFIT_TARGET
        return {
            "ok": ok,
            "n": int(len(Xs)),
            "rf_accuracy": rf_acc,
            "rf_confidence": rf_conf,
            "mlp_skipped": True,
        }

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-6,
        batch_size=min(8, len(Xs)),
        learning_rate_init=1e-2,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )
    history: list[dict[str, float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            mlp.fit(Xs, ys)
            proba = mlp.predict_proba(Xs)
            pred = mlp.predict(Xs)
            acc = float(accuracy_score(ys, pred))
            # 對二元／多類都用 true-class 機率算交叉熵。
            conf = _mean_true_class_confidence(proba, ys, mlp.classes_)
            try:
                loss = float(log_loss(ys, proba, labels=list(mlp.classes_)))
            except ValueError:
                loss = float("nan")
            history.append({"epoch": float(epoch), "accuracy": acc, "confidence": conf, "loss": loss})
            if epoch in {1, 10, 25, 50, 75, 100} or epoch == epochs:
                print(f"  Epoch {epoch:3d}: loss={loss:.4f}  acc={acc:.3f}  conf={conf:.3f}")

    final = history[-1]
    ok = final["confidence"] >= OVERFIT_TARGET and final["accuracy"] >= OVERFIT_TARGET
    print(
        f"  → MLP 最終信心 {final['confidence']:.3f} / 準確率 {final['accuracy']:.3f}："
        f"{'PASS（架構／Loss／特徵編碼大致正常）' if ok else 'FAIL（需檢查特徵或標籤）'}"
    )
    if ok and rf_conf < 0.7:
        print(
            "  提示：小樣本 MLP 可到 0.99，但線上 RandomForest 的 predict_proba 常偏軟；"
            "若正式模型在「整段訓練 session」上仍偏低，多半是類別不平衡、視窗長度不同，"
            "或樹模型本身機率未校準，不一定是前處理不一致。"
        )
    return {
        "ok": ok,
        "n": int(len(Xs)),
        "rf_accuracy": rf_acc,
        "rf_confidence": rf_conf,
        "mlp_accuracy": final["accuracy"],
        "mlp_confidence": final["confidence"],
        "mlp_loss": final["loss"],
        "epochs": epochs,
        "history_tail": history[-5:],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # 重用 train.py 的資料來源參數，再加診斷專用開關。
    base = parse_train_args([])
    parser = argparse.ArgumentParser(description="診斷姿勢品質模型前處理／標籤／過擬合")
    parser.add_argument(
        "--engine",
        choices=("pose", "mediapipe", "all"),
        default="all",
        help="預設 all（節點資料庫=/poses 與 MediaPipe 一併拉）。只查 mediapipe 容易變成 0 筆。",
    )
    parser.add_argument("--from-api", metavar="URL", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--from-json", metavar="FILE", default=None)
    parser.add_argument(
        "--save-json",
        metavar="FILE",
        default=None,
        help="搭配 --from-api：把雲端 sessions 另存成本地 JSON（之後就不必再匯出手機檔）",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="本機／雲端都沒資料時，才允許用合成 10 筆跑過擬合（預設關閉，避免誤判）",
    )
    parser.add_argument("--overfit-n", type=int, default=OVERFIT_N)
    parser.add_argument("--epochs", type=int, default=OVERFIT_EPOCHS)
    parser.add_argument("--json-out", metavar="FILE", help="把診斷結果寫成 JSON")
    # 讓未使用的 train 預設不干擾
    _ = base
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # 無 --from-api／--from-json 且未允許合成時，直接提示正確指令（桌面空檔／錯 engine 是常見原因）。
    if not args.from_api and not args.from_json and not args.allow_synthetic:
        print(
            "請用雲端帳號拉已上傳的標註（不必手機 JSON）：\n"
            "  python diagnose_quality_model.py --engine all \\\n"
            "    --from-api https://runpose-backend-production.up.railway.app \\\n"
            "    --username 你的帳號 --password 你的密碼 --save-json training_from_api.json\n"
            "若檔案是 0 byte／打不開，請先 git checkout cursor/diagnose-quality-model-b192。"
        )
        return 1

    preprocess = check_train_infer_preprocess()

    try:
        X, y, source = load_xy(args)
    except RuntimeError as exc:
        print(f"載入資料失敗：{exc}")
        return 1
    print()
    print(f"資料來源：{source}（{len(X)} 筆，{X.shape[1] if len(X) else 0} 維）")

    labels = check_label_balance(y)
    overfit = overfit_sanity_check(X, y, n=args.overfit_n, epochs=args.epochs)

    summary = {
        "preprocess_ok": bool(preprocess.get("ok")),
        "label_ratios": labels,
        "overfit_ok": bool(overfit.get("ok")),
        "data_source": source,
        "n_samples": int(len(X)),
        "preprocess": {k: v for k, v in preprocess.items() if k != "layout"},
        "overfit": overfit,
    }
    print()
    print("=" * 60)
    print("總結")
    print("=" * 60)
    print(f"  前處理一致：{'PASS' if summary['preprocess_ok'] else 'FAIL'}")
    print(f"  過擬合檢查：{'PASS' if summary['overfit_ok'] else 'FAIL'}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"  已寫入：{args.json_out}")

    return 0 if summary["preprocess_ok"] and summary["overfit_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
