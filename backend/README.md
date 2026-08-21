# Pose App 後端（帳號登入 + 姿勢節點 NoSQL 資料庫）

以 **FastAPI** 提供：

1. **帳號註冊 / 登入**（MongoDB Atlas `users` collection，密碼以 PBKDF2-SHA256 + 獨立 salt 雜湊儲存，登入回傳存取權杖）。
2. **姿勢節點資料庫（MongoDB / NoSQL）**：App 偵測後可帶「好 / 壞」標籤上傳整段節點資料，
   作為**監督式學習**的訓練資料；附 `train.py` 直接訓練二元分類模型。

## 1. 安裝

需要 Python 3.9 以上。

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 安裝並啟動 MongoDB（NoSQL，存放姿勢節點）

用 Homebrew 安裝 MongoDB Community 並啟動服務：

```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community   # 背景常駐，開機自動啟動
# 或臨時前景執行： mongod --dbpath ~/data/db
```

預設連線為 `mongodb://localhost:27017`，資料庫名 `pose`、集合 `pose_sessions`。
可用環境變數覆寫：`POSE_MONGO_URL`、`POSE_MONGO_DB`。

> 沒有 MongoDB 時，帳號登入與姿勢節點上傳皆無法運作；請設定 `POSE_MONGO_URL` 指向 Atlas 或本機 MongoDB。

## 2. 啟動

```bash
# 在 backend 資料夾、且已啟用虛擬環境的狀態下
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

啟動後：

- 本機：<http://127.0.0.1:8000>
- 互動式 API 文件（Swagger）：<http://127.0.0.1:8000/docs>
- 帳號與姿勢節點皆存 MongoDB（`users`、`pose_sessions` collections）
- 健康檢查：`GET /health/auth`、`GET /health/mongo`

> `--host 0.0.0.0` 讓區域網路內的實機 iPhone 也能連線。

## 3. 讓實機 iPhone 連到後端

1. 確保 iPhone 與這台 Mac 連在**同一個 Wi-Fi**。
2. 查出 Mac 的區域網路 IP：

   ```bash
   ipconfig getifaddr en0   # Wi-Fi；若無輸出改試 en1
   ```

3. 在 App 登入畫面的「伺服器網址」填入：`http://<Mac的IP>:8000`
   - iOS 模擬器則填 `http://127.0.0.1:8000`

## 4. API 端點

| 方法 | 路徑 | 說明 | Body |
|------|------|------|------|
| GET  | `/` | 健康檢查 | — |
| POST | `/auth/register` | 註冊（同時回傳權杖） | `{"username","password"}` |
| POST | `/auth/login` | 登入，回傳權杖 | `{"username","password"}` |
| GET  | `/auth/me` | 取得目前登入者與個人資料（需帶權杖） | Header `Authorization: Bearer <token>` |
| PUT  | `/auth/profile` | 填寫／更新個人資料（需帶權杖） | `{"display_name","gender","age","height_cm","weight_kg"}` |
| POST | `/auth/change-password` | 修改密碼（需帶權杖） | `{"current_password","new_password"}` |
| POST | `/poses` | 上傳一段姿勢節點 + 標籤（整段，需帶權杖） | `{"label":"good\|bad","frames":[...],...}` |
| GET  | `/poses` | 列出自己上傳過的 session（不含 frames） | Header `Authorization: Bearer <token>` |
| GET  | `/dataset/stats` | 資料集統計（各標籤筆數） | Header `Authorization: Bearer <token>` |
| POST | `/sessions` | 開一個即時串流 session | `{"label":"good\|bad\|null","source_label"}` |
| POST | `/sessions/{id}/frames` | 逐批 push 影格節點（串流上傳） | `{"frames":[...]}` |
| POST | `/sessions/{id}/finish` | 結束串流 session、寫入步態統計 | `{"total_steps",...}` |
| POST | `/predict` | 用訓練好的模型即時預測好/壞 | `{"frames":[...]}` |

帳號長度 3–32、密碼長度 6–128。姿勢標籤 `label` 必須是 `good` 或 `bad`（串流 session 可為 null = 不標註）。

**即時串流 vs 整段上傳**：App 偵測時改用 `/sessions` 系列「每幀即時串流」把節點逐批寫入 MongoDB；`/poses` 整段上傳保留為相容用途。

**即時預測 `/predict`**：需先用 `train.py` 訓練出 `pose_quality_model.joblib`。模型會被快取，**重新訓練後請重啟後端**才會載入新模型。

### 快速測試

```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"secret123"}'

curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"secret123"}'
```

## 5. 監督式學習：訓練姿勢品質模型

App 端在「節點資料庫」畫面選好「好 / 壞」標籤並上傳幾筆資料後，即可訓練：

```bash
cd backend
source .venv/bin/activate
python train.py            # 訓練 + 評估 + 存模型 pose_quality_model.joblib
python train.py --export   # 另外匯出 dataset.csv（特徵 + 標籤）
```

- 特徵：每個 session 的 35 個關節在所有影格的 (x,y,z) 平均與標準差（共 210 維）。
- 模型：RandomForest 二元分類（good / bad）。
- 樣本太少或只有單一標籤時會提示先補資料（建議好、壞各至少數筆）。

## 6. 正式環境注意

- `SECRET_KEY` 請改用環境變數 `POSE_SECRET_KEY`，不要使用預設值。
- 對外服務請改走 HTTPS（搭配反向代理，如 Nginx / Caddy）。
- iOS 端 Release 版連 HTTPS 雲端後端，見 **`DEPLOY.md`**（出門也能用）。

## 7. 雲端部署（出門也能用）

完整步驟見 **[DEPLOY.md](./DEPLOY.md)**：MongoDB Atlas + Railway，並在 App 的 `Info.plist` 設定 `PoseServerBaseURL`。
