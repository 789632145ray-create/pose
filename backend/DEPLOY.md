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
            └── pose_sessions（姿勢節點 / 訓練標籤）

iPhone 本機 Realm Database（pose.realm，純本機、不同步雲端）
    ├── 姿勢節點 session（取代舊版 SQLite）
    └── 分析摘要歷史（取代舊版 JSON）
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

1. 到 [Railway](https://railway.app) 用 GitHub 登入
2. **New Project** → **Deploy from GitHub repo** → 選 **`runpose-backend`**
3. **Root Directory**：repo 根目錄就是 `main.py` 時 **留空**
4. **Variables** 新增：

   | 變數 | 值 |
   |------|-----|
   | `POSE_SECRET_KEY` | 隨機長字串（`openssl rand -hex 32`） |
   | `POSE_MONGO_URL` | Atlas 連線字串 |
   | `POSE_MONGO_DB` | `pose` |
   | `POSE_RELOAD` | `0` |

5. **Networking** → **Generate Domain**
6. 部署完成後：
   - `GET /` → `{"status":"ok","service":"pose-auth"}`
   - `GET /health/auth` → `{"status":"ok","storage":"mongodb_atlas",...}`
   - `GET /health/mongo` → MongoDB 連線正常

### 上傳訓練模型（品質辨識）

```bash
cd backend
source .venv/bin/activate
python train.py   # 產生 pose_quality_model.joblib
```

把 `pose_quality_model.joblib` commit 進 repo 或上傳到 Railway 容器 `/app/`，然後重啟服務。

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

**Q: Build 失敗「Failed to build an image」？**  
A: Root Directory 設錯。選 `runpose-backend` repo 時 Root Directory **留空**。
