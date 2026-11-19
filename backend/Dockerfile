FROM python:3.11-slim

WORKDIR /app

# scikit-learn 需要編譯/執行時函式庫；ca-certificates + openssl 供 Atlas TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    ca-certificates \
    openssl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py train.py mongo_util.py video_result_aggregator.py ./
COPY *.joblib ./
# 若尚無 mediapipe_quality_model.joblib，/mediapipe/predict 會回 503；訓練後 commit 該檔即可。

ENV POSE_RELOAD=0

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
