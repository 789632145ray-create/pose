# 雲端部署指南（出門也能用）

把後端部署到公開網路後，App 用 **4G / 5G / 任何 Wi‑Fi** 都能登入、上傳節點、做品質辨識，不再需要和 Mac 在同一區域網路。

## 架構

```
iPhone（任何地方）
    │  HTTPS（登入 / 上傳 / 辨識）
    ▼
Railway / Render（FastAPI 容器）
    └── MongoDB Atlas
            ├── users（帳號，PBKDF2 雜湊密碼）
            ├── pose_sessions（QuickPose／自訓模型節點 / 訓練標籤）
            └── mediapipe_sessions（MediaPipe 骨架，獨立 collection）

iPhone 本機 Realm Database（純本機、不同步雲端）
    ├── pose.realm：QuickPose／自訓模型節點 + 分析摘要
    └── mediapipe.realm：MediaPipe 骨架 session
```

> **關於 App Services**：MongoDB Atlas App Services 已於 **2025 年 9 月 30 日**正式下線（EOL），Atlas 介面中已無法建立。本專案帳號改由 **FastAPI + MongoDB `users` collection** 管理；Realm 僅作本機資料庫使用。

## 第一步：MongoDB Atlas（免費）

1. 到 [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) 註冊
2. 建立 **Free M0** 叢集
3. **Database Access** → 新增使用者（記下帳密）
4. **Network Access** → Add IP Address → **Allow Access from Anywhere**（`0.0.0.0/0`）
5. **Connect** → Drivers → 複製連線字串，例如：
   ```
   mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

## 第二步：部署到 Railway（建議）

Railway **只跑後端 API**，專案列表不會出現 iOS App（App 用 Xcode 裝在手機）。

### Railway Projects 是空的？

先確認這兩件事：

1. **舊服務可能還在跑。** App 現在連的 `https://runpose-backend-production.up.railway.app` 若 `/` 回 `{"status":"ok"}`，代表雲端沒掛，只是你這個 Railway 帳號／工作區看不到它。
2. 打開 [Railway dashboard](https://railway.app/dashboard) 左上角 **workspace**，切換個人帳號與任何 Team。登錯 GitHub / Google 帳號時，列表會是空的。

看不到舊專案就重建一個（資料在 MongoDB Atlas，不在 Railway）：

1. 到 [Railway](https://railway.app) 用 **同一個 GitHub** 登入
2. **New Project** → **Deploy from GitHub repo** → 選 **`789632145ray-create/pose`**
3. 若沒看到 repo：Account Settings → GitHub → 勾選 `pose`
4. Branch 選 **`cursor/add-runpose-backend-dd14`**（合併進 `main` 後再改 `main`）
5. **Root Directory 留空**（根目錄 `Dockerfile` 會去抓 `backend/`）
6. **Variables** 新增：

   | 變數 | 值 |
   |------|-----|
   | `POSE_SECRET_KEY` | 隨機長字串（`openssl rand -hex 32`）。換了金鑰後手機要重新登入 |
   | `POSE_MONGO_URL` | Atlas **完整**連線字串（到 Atlas → Connect 複製，不要用 `mongodb+srv://...`） |
   | `POSE_MONGO_DB` | `pose` |
   | `POSE_RELOAD` | `0` |

7. 服務開好後：**Settings → Networking → Generate Domain**
8. 瀏覽器打開：
   - `https://你的新網址.up.railway.app/` → `{"status":"ok","service":"pose-auth"}`
   - `https://你的新網址.up.railway.app/health/mongo` → Mongo 正常
9. 把新網址填進 `pose/Info.plist` 的 `PoseServerBaseURL`，重新編譯 App

用**同一個 Atlas 連線**，原本的帳號與標籤資料還在。換了新資料庫就會變空，要重新註冊、重新標。

### 上傳訓練模型（品質辨識）

手機上傳的標籤在 **Atlas**，本機 `train.py` 預設連 `mongodb://localhost:27017`（通常是空的）。請用和 Railway 相同的連線，或從雲端 API 拉資料：

```bash
cd backend
source .venv/bin/activate

# 方式 A：連 Railway 同一個 Atlas（必須複製完整字串，不要用 mongodb+srv://...）
export POSE_MONGO_URL="mongodb+srv://使用者:密碼@cluster0.xxxx.mongodb.net/"
python train.py --engine mediapipe          # mediapipe_quality_model.joblib
python train.py                             # pose_quality_model.joblib

# 方式 B：App 資料庫「匯出 JSON」傳到 Mac
python train.py --engine mediapipe --from-json mediapipe_training_export.json

# 方式 C：用 App 帳號從雲端匯出（需已部署 /dataset/export）
python train.py --engine mediapipe \
  --from-api https://runpose-backend-production.up.railway.app \
  --username 你的帳號 --password 你的密碼
```

把對應的 `.joblib` commit 進 repo 或上傳到 Railway 容器 `/app/`，然後重啟服務。

## 第三步：設定 iOS App

在 `pose/Info.plist` 填入 Railway HTTPS 網址：

```xml
<key>PoseServerBaseURL</key>
<string>https://你的-railway-網址.up.railway.app</string>
```

> **Debug 模擬器**仍連 `http://127.0.0.1:8000` 本機後端。

## 第四步：註冊帳號

在 App 內 **註冊** 新帳號（存於 Atlas `users` collection）。

本機若仍有舊版 `pose.sqlite3` / `pose_summaries.json`，首次啟動會自動匯入 Realm 並備份為 `.bak`。

---

## 其他平台

### Render

Root Directory: `backend`，Environment: **Docker**，環境變數同上。

### 本機 Docker 測試

```bash
cd backend
docker build -t pose-api .
docker run -p 8000:8000 \
  -e POSE_SECRET_KEY=dev-secret \
  -e POSE_MONGO_URL="mongodb+srv://..." \
  pose-api
```

---

## 常見問題

**Q: Atlas 裡找不到 App Services？**  
A: 已於 2025/9/30 下線，MongoDB 不再提供此功能。本 App 帳號走 FastAPI，不需 App Services。

**Q: 上傳／辨識回 401？**  
A: 重新登入；確認 Railway 已設定 `POSE_SECRET_KEY` 且 `POSE_MONGO_URL` 正確。

**Q: 沒網路能用嗎？**  
A: 已登入者可做本機偵測與看 Realm 歷史；登入、上傳、AI 辨識需要網路。

**Q: Railway Projects 裡面沒有東西？**  
A: 多半是登錯帳號或左上角 workspace 不對。舊網址若還能開，服務還在，只是這個帳號看不到。看不到就依上面「第二步」用 `pose` repo 新建，Variables 接回同一個 Atlas。

**Q: Build 失敗「Failed to build an image」？**  
A: 選 `pose` repo 時 Root Directory **留空**（用根目錄 `Dockerfile`）。若改連舊的 `runpose-backend` 且 `main.py` 在根目錄，Root Directory 也是留空。
