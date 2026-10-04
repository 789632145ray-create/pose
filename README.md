# pose

iOS App 與後端都在這個 repo，**不要搬去 `runpose-backend`**。

- App：`pose.xcodeproj`（Xcode 打開這個）
- 後端：`backend/`（Railway 只部署這層）

Railway 專案列表是空的時，用這個 repo 重建即可：**New Project → GitHub → `pose`**，Root Directory 留空。步驟見 [backend/DEPLOY.md](./backend/DEPLOY.md)。

本機訓練：

```bash
cd backend
source .venv/bin/activate
python train.py --engine mediapipe --from-json mediapipe_training_export.json
```
