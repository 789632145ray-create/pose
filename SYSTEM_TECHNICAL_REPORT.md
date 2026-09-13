# Pose 姿勢偵測系統 — 系統與技術報告

| 項目 | 內容 |
|------|------|
| 系統名稱 | Pose 姿勢偵測 App |
| 平台 | iOS 17+（SwiftUI），需實機（相機／GPU） |
| 後端 | FastAPI + MongoDB Atlas，部署於 Railway |
| 本機資料 | Realm（`pose.realm`、`mediapipe.realm`） |
| 骨架來源 | QuickPose SDK（內建 MediaPipe BlazePose Full） |
| 文件版本 | 對應分支 `cursor/add-runpose-backend-dd14` |
| 主要程式庫 | `https://github.com/789632145ray-create/pose` |

---

## 1. 系統概述

本系統是一套**走路／跑步姿勢偵測與教練 App**。使用者以 iPhone 相機或本機影片擷取全身骨架，系統即時計算步態、對齊與活動姿勢核心，並可把節點標成「好／壞」上傳雲端，供監督式學習訓練姿勢品質模型。

系統分成兩端：

1. **使用者端 App（iOS）**：登入、選引擎、選走路／跑步、即時骨架、規則建議、本機節點庫、歷史摘要。
2. **雲端後端（Railway）**：帳號、節點上傳、即時串流、RandomForest `/predict`。

影片分析在**手機本機**完成，不需把影片傳到 Railway。雲端只收關節座標與標籤。

---

## 2. 系統架構

```
┌──────────────────────────────────────────────────────────────────┐
│                         使用者端 iOS App                          │
│  登入／會員 ─ 偵測畫面 ─ 走路／跑步核心 ─ 本機 Realm ─ 歷史摘要   │
│         │ 相機／影片                                              │
│         ▼                                                        │
│  QuickPose SDK（BlazePose Full，35 關節／幀）                    │
│         │                                                        │
│         ├─ 規則建議（對齊、走路核心、跑步核心、個人化步態）       │
│         ├─ 步數／步頻（低通 → 骨盆歸一 → 腳踝峰谷）               │
│         └─ 自訓模型：節點串流 → POST /predict                    │
└─────────┼────────────────────────────────────────────────────────┘
          │ HTTPS（Bearer Token）
          ▼
┌──────────────────────────────────────────────────────────────────┐
│  FastAPI（Railway）                                              │
│  /auth/*   /poses   /mediapipe/poses   /sessions/*   /predict    │
│         │                                                        │
│         ▼                                                        │
│  MongoDB Atlas：users / pose_sessions / mediapipe_sessions       │
│  模型檔：pose_quality_model.joblib（RandomForest，210 維）       │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 端到端資料流

```
相機或本機影片
    → QuickPose 影格回呼（status, overlay, landmarks）
    → PoseNodeExtractor（35 關節）
    → 本機 Realm 寫入（依引擎分流）
    → PoseAdvice + WalkingFormAdvisor 或 RunningFormAdvisor
    → PoseAnalysisPipeline（步數、問題累積、摘要）
    → 畫面 HUD／建議
    →（自訓模型）PoseStreamClient 串流 + /predict
    →（暫停後）節點庫標好／壞 → POST /poses（或 /mediapipe/poses，404 則回退 /poses）
    → train.py 讀 Mongo → 產出 joblib → 後端 /predict
```

---

## 3. 技術堆疊

| 層級 | 技術 | 用途 |
|------|------|------|
| App UI | SwiftUI、iOS 17 | 偵測畫面、選單、Sheet |
| 骨架 SDK | QuickPoseCore / QuickPoseSwiftUI / QuickPoseMP-full | 相機、影片模擬相機、BlazePose Full |
| 本機 DB | Realm Swift | session、影格節點、歷史摘要 |
| 帳號儲存 | Keychain | Bearer Token |
| 後端 | Python 3.9+、FastAPI、Uvicorn | REST API |
| 雲端 DB | MongoDB Atlas | 使用者、訓練 session |
| ML | scikit-learn RandomForest | 好／壞二元分類 |
| 部署 | Docker、Railway | 正式 HTTPS API |
| 訓練 | `backend/train.py`（本機／Mac） | 從 Mongo 抽 210 維特徵並訓練 |

---

## 4. 功能模組

### 4.1 使用者端 App

| 功能 | 說明 | 主要檔案 |
|------|------|----------|
| 路由 | 未登入 → 登入；登入後 → 偵測 | `RootView.swift` |
| 帳號 | 註冊／登入／登出、改密碼、個人資料 | `AuthManager.swift`, `LoginView.swift`, `MemberProfileSheet.swift` |
| 引擎 | QuickPose／MediaPipe／自訓模型，皆走同一條完整管線 | `PoseAssessmentEngine.swift`, `PoseDetectionShellView.swift` |
| 活動 | 走路／跑步（著地與軀幹標準不同） | `GaitActivityMode.swift` |
| 偵測 | 相機授權、倒數、開始／暫停、影片匯入 | `PoseDetectionView.swift` |
| 骨架顯示 | QuickPose overlay；MediaPipe 另畫薄荷色關節連線 | `MediaPipeSkeletonOverlay.swift` |
| 選單 | 歷史、會員、引擎、登出（放在左上資訊卡內，避開狀態列熱區） | `AppMenuSheet.swift` |

### 4.2 資料蒐集

每幀擷取 35 個身體關節：`nose`、`shoulder_mid`、`hip_mid`，以及左右眼／耳／口、肩、肘、腕、指、髖、膝、踝、跟、足尖。

單一節點：

```text
joint, x, y, z, visibility, presence
```

- QuickPose、自訓模型 → `pose.realm`
- MediaPipe → `mediapipe.realm`
- 寫入在背景序列佇列，避免卡住影格回呼

### 4.3 特徵擷取與步態

```
Landmarks → 5 幀低通 → 骨盆中心歸一 → 左右腳踝 y 峰／谷 → 步事件
```

同時累積姿勢問題次數，供整段摘要。

Session 級 ML 特徵（後端）：35 關節 × (x, y, z) 的平均與標準差 = **210 維**。

### 4.4 AI 姿勢品質辨識（自訓模型）

| 項目 | 說明 |
|------|------|
| 模型 | `RandomForestClassifier` |
| 檔案 | `backend/pose_quality_model.joblib` |
| 標籤 | `good` / `bad` |
| 即時 API | `POST /predict` |
| App 呼叫 | 僅「自訓模型」引擎會串流並預測 |

QuickPose、MediaPipe **不呼叫** `/predict`，只做裝置上規則建議。

### 4.5 健康／姿勢評估與建議

即時建議由三層合併：

1. **對齊**：肩高低、骨盆傾斜、軀幹側傾、頭部中線  
2. **活動核心**：走路或跑步（見第 6 節）  
3. **個人化步態**：依年齡、BMI、身高調整步幅、步頻、拖步、髖伸展等提示  

暫停時顯示該活動的核心條文與休息建議。結束後寫入歷史摘要。

---

## 5. 三種偵測引擎

三種引擎共用**同一個已啟動的 QuickPose Full 工作階段**畫骨架。切換時不 `stop()`／`start()`，只改分析路徑與資料庫，避免 overlay 線條只在第一個引擎出現。

| 引擎 | 骨架 | 建議 | 本機庫 | 雲端 |
|------|------|------|--------|------|
| QuickPose | BlazePose Full overlay | 規則（對齊 + 走路／跑步） | `pose.realm` | 標好／壞 → `/poses` |
| MediaPipe | 同上，另畫關節連線 | 規則（同上） | `mediapipe.realm` | 先試 `/mediapipe/poses`，404 則 `/poses` |
| 自訓模型 | 同上 | 以 `/predict` 為主 | `pose.realm` | 串流 `/sessions` + `/predict` |

正式 Railway 若尚未部署本 repo 的 `/mediapipe/poses`，App 會自動改傳到既有 `/poses`，不必新建專案。

---

## 6. 走路／跑步姿勢核心

底部可切換「走路」或「跑步」。兩套標準分開，避免走路「腳跟先著地」與跑步「中足著地」互相打架。

### 6.1 走路

1. 手臂自然下垂，隨著**對側腳**前後擺動，幅度不宜過大。  
2. **腳跟先著地**，力量平穩過渡到腳掌，最後由**腳尖蹬地**前進。

即時可偵測：手臂未下垂、擺臂過大、同側擺臂、腳尖先著地、後腳缺少蹬地。

### 6.2 跑步

1. 由**腳踝發力**，身體**整體**微前傾約 10 度，不是從腰彎；用地心引力把重心前移。  
2. **微收腹、穩定骨盆**，避免左右搖晃或上下起伏過大。  
3. 腳掌落在**重心正下方**，避免過度跨步；建議**中足著地**，用足弓緩衝膝蓋。  
4. 手肘約 **90 度**、輕握拳，手臂前後擺，**不要越過身體中線**。

即時可偵測：從腰往前折、前傾不足／過大、骨盆不穩、過度跨步、跑步用腳跟著地、手肘角度偏差、擺臂過中線。

---

## 7. 資料庫設計

### 7.1 本機 Realm

| 檔案 | 內容 |
|------|------|
| `pose.realm` | QuickPose／自訓模型的 session、影格節點、歷史摘要 |
| `mediapipe.realm` | 僅 MediaPipe session 與節點 |

Session 含來源（相機／影片檔名）、起訖時間、幀數、節點數、左右步數。每個影格嵌入該幀全部關節。

### 7.2 雲端 MongoDB

| Collection | 內容 |
|------------|------|
| `users` | 帳號、PBKDF2-SHA256 密碼雜湊、個人資料 |
| `pose_sessions` | 標註或串流的骨架 session |
| `mediapipe_sessions` | MediaPipe 專用 session（後端已支援；舊部署可能尚未上線） |

---

## 8. 後端 API

正式網址設定於 `Info.plist` 的 `PoseServerBaseURL`（目前為 Railway HTTPS）。Debug 模擬器改連 `http://127.0.0.1:8000`。

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 健康檢查 |
| GET | `/health/mongo`、`/health/auth`、`/health/model` | 依賴檢查 |
| POST | `/auth/register`、`/auth/login` | 註冊／登入，回傳權杖 |
| GET | `/auth/me` | 目前使用者 |
| PUT | `/auth/profile` | 姓名、性別、年齡、身高、體重 |
| POST | `/auth/change-password` | 改密碼 |
| POST / GET | `/poses` | 整段標好／壞上傳與列表 |
| GET | `/dataset/stats` | 資料集統計 |
| POST / GET | `/mediapipe/poses` | MediaPipe 獨立上傳（舊雲端 404 時 App 回退） |
| POST | `/sessions`、`/sessions/{id}/frames`、`/sessions/{id}/finish` | 即時串流 |
| POST | `/predict` | RandomForest 好／壞預測 |

權杖：HMAC 簽章，預設 7 天。密碼：PBKDF2-SHA256、200,000 次、每帳號獨立 salt。正式環境應設定 `POSE_SECRET_KEY`。

---

## 9. 介面與操作

1. 登入或註冊。  
2. 首次可選偵測引擎；之後可在底部三鍵切換。  
3. 選**走路**或**跑步**。  
4. 允許相機後按「開始」，倒數結束開始寫入資料庫。  
5. 或用「選影片／上傳影片／檔案」做本機分析。  
6. 左上資訊卡內的三橫線開啟選單（歷史、會員、登出）。  
7. 暫停後可開節點庫，標好／壞並上傳。

選單不放在螢幕右上角，避免疊到 iOS 狀態列與控制中心熱區導致無法點擊。

---

## 10. 部署與執行

### 10.1 App

- 必須用**實體 iPhone**（模擬器無相機，QuickPose 需 GPU）。  
- Xcode 開啟專案，確認 Bundle ID 已綁 QuickPose 金鑰。  
- 實機 Debug 連正式 Railway；模擬器連本機 8000。

### 10.2 後端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

訓練：

```bash
python train.py            # 寫入 pose_quality_model.joblib
python train.py --export   # 另存 dataset.csv
```

重新訓練後需**重啟後端**才會載入新模型。雲端步驟見 `backend/DEPLOY.md`。

---

## 11. 測試

`poseTests` 覆蓋：

- 三種引擎的職責（規則 vs `/predict`、資料庫分流）  
- 舊版引擎字串相容  
- MediaPipe 骨架連線使用已知關節  
- 切換引擎不重啟 QuickPose runtime  
- 走路核心：對側擺臂、腳跟→腳尖  
- 跑步核心：前傾、中足、手肘 90 度、不越過中線  

UI 需實機目視：骨架線、走路／跑步建議、選單可點。

---

## 12. 限制與已知事項

| 項目 | 說明 |
|------|------|
| 實機 | 相機與 QuickPose 無法在模擬器完整驗證 |
| 視角 | 前傾、跨步、中線擺臂在側拍或正拍較準；斜拍誤差較大 |
| 影片 + 部分 overlay | 曾避免 `.showPoints()` 以免 SimulatedCamera 不穩；MediaPipe 改由 App 自繪節點線 |
| 正式 `/mediapipe/poses` | 需部署本 repo 的 `backend/` 才會出現；未部署時 App 回退 `/poses` |
| 模型品質 | 好／壞樣本太少或只有單類時，RandomForest 不可靠 |
| 安全 | 開發預設 `POSE_SECRET_KEY` 不可用於正式環境 |

---

## 13. 主要原始碼對照

| 檔案 | 職責 |
|------|------|
| `pose/PoseDetectionView.swift` | 偵測主畫面、相機／影片、HUD、選單位置 |
| `pose/PoseDetectionShellView.swift` | 引擎與走路／跑步切換 |
| `pose/PoseAssessmentEngine.swift` | 三引擎定義 |
| `pose/WalkingFormAdvisor.swift` | 走路核心 |
| `pose/RunningFormAdvisor.swift` | 跑步核心 |
| `pose/GaitPersonalization.swift` | BMI／年齡個人化步態 |
| `pose/PoseAnalysisPipeline.swift` | 步數、問題碼、摘要 |
| `pose/PoseDatabase.swift` | 節點擷取與 Realm 寫入 |
| `pose/PoseStreamClient.swift` | 串流、預測、標註上傳 |
| `pose/MediaPipeSkeletonOverlay.swift` | MediaPipe 關節線 |
| `backend/main.py` | FastAPI |
| `backend/train.py` | 特徵與訓練 |
| `backend/mongo_util.py` | Mongo 連線 |
| `ARCHITECTURE.md` | 八大模組架構圖（較早版本，細節以本報告為準） |

---

## 14. 結論

系統已具備完整閉環：**裝置上骨架偵測 → 走路／跑步教練 → 本機節點庫 → 雲端標註與品質模型**。三種引擎共用同一骨架工作階段，活動模式分開評估，後端以 FastAPI + Mongo 支援帳號與訓練資料。後續若要強化，優先項目是把本 repo 後端部署到既有 Railway（讓 `/mediapipe/poses` 上線）、補齊好／壞訓練樣本，以及針對側拍視角校正前傾與著地判斷。
