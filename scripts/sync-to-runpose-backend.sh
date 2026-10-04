#!/usr/bin/env bash
# 把這個 pose repo 的 iOS App + 後端同步到本機的 runpose-backend 目錄。
# 後端會放在 runpose-backend 根目錄（Railway 需要 main.py 在根目錄）。
#
# 用法：
#   git clone https://github.com/789632145ray-create/runpose-backend.git ~/runpose-backend
#   ./scripts/sync-to-runpose-backend.sh ~/runpose-backend
#   cd ~/runpose-backend
#   git checkout -b cursor/move-ios-app-dd14
#   git add -A
#   git commit -m "Bring the iOS app back into runpose-backend."
#   git push -u origin cursor/move-ios-app-dd14

set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:-}"

if [[ -z "$DST" || ! -d "$DST/.git" ]]; then
  echo "請指定本機 runpose-backend 的 git 目錄，例如："
  echo "  $0 ~/runpose-backend"
  exit 1
fi

DST="$(cd "$DST" && pwd)"

if [[ ! -f "$DST/main.py" && ! -f "$DST/railway.toml" ]]; then
  echo "目標不像是 runpose-backend（找不到 main.py 或 railway.toml）：$DST"
  exit 1
fi

echo "來源：$SRC"
echo "目標：$DST"

# 保留 Railway 上已部署的訓練模型
if [[ -f "$DST/pose_quality_model.joblib" ]]; then
  cp -a "$DST/pose_quality_model.joblib" /tmp/pose_quality_model.joblib.prod
fi

cp -a "$SRC/backend/main.py" \
      "$SRC/backend/train.py" \
      "$SRC/backend/mongo_util.py" \
      "$SRC/backend/test_train.py" \
      "$SRC/backend/requirements.txt" \
      "$SRC/backend/Dockerfile" \
      "$SRC/backend/railway.toml" \
      "$SRC/backend/.dockerignore" \
      "$DST/"
cp -a "$SRC/backend/README.md" "$DST/BACKEND.md"
cp -a "$SRC/backend/DEPLOY.md" "$DST/DEPLOY.md"

if [[ -f /tmp/pose_quality_model.joblib.prod ]]; then
  cp -a /tmp/pose_quality_model.joblib.prod "$DST/pose_quality_model.joblib"
fi

rm -rf "$DST/pose" "$DST/pose.xcodeproj" "$DST/poseTests" "$DST/poseUITests" "$DST/scripts"
cp -a "$SRC/pose" "$DST/"
mkdir -p "$DST/pose.xcodeproj"
cp -a "$SRC/pose.xcodeproj/project.pbxproj" "$DST/pose.xcodeproj/"
cp -a "$SRC/pose.xcodeproj/project.xcworkspace" "$DST/pose.xcodeproj/"
rm -rf "$DST/pose.xcodeproj/xcuserdata" "$DST/pose.xcodeproj/project.xcworkspace/xcuserdata"
cp -a "$SRC/poseTests" "$DST/"
cp -a "$SRC/poseUITests" "$DST/"
mkdir -p "$DST/scripts"
cp -a "$SRC/scripts/md_to_docx.py" "$DST/scripts/" 2>/dev/null || true
cp -a "$SRC/ARCHITECTURE.md" "$SRC/SYSTEM_TECHNICAL_REPORT.md" "$DST/"
if [[ -f "$SRC/Pose系統與技術報告.docx" ]]; then
  cp -a "$SRC/Pose系統與技術報告.docx" "$DST/"
fi

# 根目錄說明：App + 後端同一個 repo
cat > "$DST/README.md" <<'EOF'
# RunPose（iOS App + 後端）

這個 repo 同時放 **iOS App** 和 **Railway 後端**。後端必須留在**根目錄**（`main.py`），Railway 才會繼續部署。

```
pose/ pose.xcodeproj/     iOS App（Xcode 打開 pose.xcodeproj）
poseTests/ poseUITests/
main.py train.py          FastAPI 後端（Railway Root Directory 留空）
DEPLOY.md                 雲端部署
```

## iOS App

用 Xcode 打開 `pose.xcodeproj`。

## 後端（本機）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 訓練

```bash
source .venv/bin/activate
python train.py --engine mediapipe --from-json mediapipe_training_export.json
```
EOF

# 文件裡的 cd backend 改成根目錄
python3 - <<PY
from pathlib import Path
for name in ("DEPLOY.md", "BACKEND.md", "SYSTEM_TECHNICAL_REPORT.md"):
    path = Path("$DST") / name
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    text = text.replace("cd backend\n", "")
    text = text.replace("Root Directory: `backend`", "Root Directory: **留空**（main.py 在 repo 根目錄）")
    path.write_text(text, encoding="utf-8")
PY

rm -f "$DST/.DS_Store"
echo "同步完成。接下來在 $DST："
echo "  git checkout -b cursor/move-ios-app-dd14"
echo "  git add -A && git status"
echo "  git commit -m 'Bring the iOS app back into runpose-backend.'"
echo "  git push -u origin cursor/move-ios-app-dd14"
