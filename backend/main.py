"""
Pose App 後端：帳號註冊 / 登入 API。

- 帳號：MongoDB Atlas `users` collection（與姿勢節點同一叢集）。
- 密碼：以 PBKDF2-SHA256 + 每個帳號獨立 salt 雜湊後儲存，絕不存明碼。
- 登入：成功後簽發 HMAC 簽章存取權杖，iOS 端帶 Authorization: Bearer <token>。

啟動方式見同資料夾 README.md。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# MongoDB（NoSQL，存放姿勢節點資料，供監督式學習使用）
from bson import ObjectId
from mongo_util import make_mongo_client, ping_mongo
from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

# 權杖簽章金鑰：正式環境請改用環境變數，不要寫死在程式碼。
SECRET_KEY = os.environ.get("POSE_SECRET_KEY", "dev-only-change-me-in-production")

# 權杖有效時間（秒），預設 7 天。
TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60

PBKDF2_ITERATIONS = 200_000

# MongoDB 連線設定（NoSQL）。預設連本機；可用環境變數覆寫。
MONGO_URL = os.environ.get("POSE_MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("POSE_MONGO_DB", "pose")

# 監督式學習的合法標籤（姿勢品質：好 / 壞）。
VALID_LABELS = {"good", "bad"}


# MongoDB（帳號 + 姿勢節點，皆存 Atlas）
mongo_client = make_mongo_client(MONGO_URL)
mongo_db = mongo_client[MONGO_DB_NAME]
users_col = mongo_db["users"]
pose_sessions = mongo_db["pose_sessions"]


def init_mongo_indexes() -> None:
    users_col.create_index("username", unique=True)


# ---------------------------------------------------------------------------
# 密碼雜湊
# ---------------------------------------------------------------------------


def hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return dk.hex()


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


# ---------------------------------------------------------------------------
# 存取權杖（HMAC 簽章，header.payload.signature）
# ---------------------------------------------------------------------------


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_token(username: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"sub": username, "iat": now, "exp": now + TOKEN_TTL_SECONDS}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"


def decode_token(token: str) -> str:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="權杖格式錯誤")

    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_decode(signature_b64), expected_sig):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="權杖簽章無效")

    payload = json.loads(_b64url_decode(payload_b64))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="權杖已過期，請重新登入")

    return str(payload["sub"])


def current_user(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少授權權杖")
    token = authorization.split(" ", 1)[1].strip()
    return decode_token(token)


# ---------------------------------------------------------------------------
# 請求 / 回應模型
# ---------------------------------------------------------------------------


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class MeResponse(BaseModel):
    username: str
    display_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    profile_completed: bool = False


class ProfileUpdateIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=32)
    gender: str = Field(description="male / female / other")
    age: int = Field(ge=1, le=120)
    height_cm: float = Field(ge=50, le=250)
    weight_kg: float = Field(ge=20, le=300)


class ChangePasswordIn(BaseModel):
    current_password: str = Field(min_length=6, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


# ---------------------------------------------------------------------------
# 應用程式
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_mongo_indexes()
    yield


app = FastAPI(title="Pose Auth API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "pose-auth"}


@app.get("/health/mongo")
def health_mongo() -> dict:
    ok, msg = ping_mongo(mongo_client)
    if ok:
        return {"status": "ok", "mongo": msg}
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)


@app.get("/health/auth")
def health_auth() -> dict:
    ok, msg = ping_mongo(mongo_client)
    if not ok:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
    return {
        "status": "ok",
        "storage": "mongodb_atlas",
        "users": users_col.count_documents({}),
    }


@app.get("/health/model")
def health_model() -> dict:
    try:
        bundle = get_model()
        classes = [int(c) for c in bundle["model"].classes_]
        return {"status": "ok", "model": "pose_quality_model.joblib", "classes": classes}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@app.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(creds: Credentials) -> TokenResponse:
    username = creds.username.strip().lower()
    salt = secrets.token_bytes(16)
    password_hash = hash_password(creds.password, salt)

    try:
        users_col.insert_one(
            {
                "username": username,
                "password_hash": password_hash,
                "salt": salt.hex(),
                "created_at": time.time(),
            }
        )
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="此帳號已存在，請改用其他帳號或直接登入",
        )

    return TokenResponse(access_token=create_token(username), username=username)


@app.post("/auth/login", response_model=TokenResponse)
def login(creds: Credentials) -> TokenResponse:
    username = creds.username.strip().lower()
    row = users_col.find_one({"username": username})

    if row is None or not verify_password(creds.password, row["salt"], row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="帳號或密碼錯誤")

    return TokenResponse(access_token=create_token(username), username=username)


def _profile_completed(row: dict) -> bool:
    for key in ("display_name", "gender", "age", "height_cm", "weight_kg"):
        if row.get(key) in (None, ""):
            return False
    return True


def _me_response(row: dict) -> MeResponse:
    return MeResponse(
        username=row["username"],
        display_name=row.get("display_name"),
        gender=row.get("gender"),
        age=row.get("age"),
        height_cm=row.get("height_cm"),
        weight_kg=row.get("weight_kg"),
        profile_completed=_profile_completed(row),
    )


@app.get("/auth/me", response_model=MeResponse)
def me(username: str = Depends(current_user)) -> MeResponse:
    row = users_col.find_one({"username": username})
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="找不到使用者")
    return _me_response(row)


@app.put("/auth/profile", response_model=MeResponse)
def update_profile(payload: ProfileUpdateIn, username: str = Depends(current_user)) -> MeResponse:
    gender = payload.gender.strip().lower()
    if gender not in {"male", "female", "other"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="gender 必須是 male、female 或 other")

    display_name = payload.display_name.strip()
    if not display_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="姓名不可為空")

    users_col.update_one(
        {"username": username},
        {
            "$set": {
                "display_name": display_name,
                "gender": gender,
                "age": payload.age,
                "height_cm": payload.height_cm,
                "weight_kg": payload.weight_kg,
                "profile_updated_at": time.time(),
            }
        },
    )
    row = users_col.find_one({"username": username})
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="找不到使用者")
    return _me_response(row)


@app.post("/auth/change-password")
def change_password(payload: ChangePasswordIn, username: str = Depends(current_user)) -> dict:
    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="新密碼不可與目前密碼相同")

    row = users_col.find_one({"username": username})
    if row is None or not verify_password(payload.current_password, row["salt"], row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="目前密碼錯誤")

    salt = secrets.token_bytes(16)
    new_hash = hash_password(payload.new_password, salt)
    users_col.update_one(
        {"username": username},
        {"$set": {"password_hash": new_hash, "salt": salt.hex()}},
    )

    return {"ok": True}


# ---------------------------------------------------------------------------
# 姿勢節點（NoSQL / MongoDB）：監督式學習資料收集
# ---------------------------------------------------------------------------


class PoseNode(BaseModel):
    joint: str
    x: float
    y: float
    z: float
    visibility: float
    presence: float


class PoseFrame(BaseModel):
    frame_index: int
    timestamp: float
    nodes: List[PoseNode]


class PoseSessionIn(BaseModel):
    """一次偵測的完整節點資料 + 監督式學習標籤。"""

    label: str = Field(description="姿勢品質標籤：good 或 bad")
    source_label: str = ""
    total_steps: int = 0
    left_steps: int = 0
    right_steps: int = 0
    avg_cadence_bpm: Optional[float] = None
    frames: List[PoseFrame]


class PoseUploadResponse(BaseModel):
    id: str
    frame_count: int
    node_count: int


class PoseSessionSummary(BaseModel):
    id: str
    label: str
    source_label: str
    frame_count: int
    node_count: int
    total_steps: int
    created_at: float


@app.post("/poses", response_model=PoseUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_pose_session(session: PoseSessionIn, username: str = Depends(current_user)) -> PoseUploadResponse:
    label = session.label.strip().lower()
    if label not in VALID_LABELS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="label 必須是 good 或 bad")
    if not session.frames:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="frames 不可為空")

    doc = session.model_dump()
    doc["label"] = label
    doc["user"] = username
    doc["created_at"] = time.time()
    doc["frame_count"] = len(session.frames)
    doc["node_count"] = sum(len(f.nodes) for f in session.frames)

    try:
        result = pose_sessions.insert_one(doc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"無法寫入資料庫：{exc}")

    return PoseUploadResponse(id=str(result.inserted_id), frame_count=doc["frame_count"], node_count=doc["node_count"])


@app.get("/poses", response_model=List[PoseSessionSummary])
def list_pose_sessions(username: str = Depends(current_user)) -> List[PoseSessionSummary]:
    try:
        cursor = pose_sessions.find(
            {"user": username},
            {"frames": 0},  # 列表不回傳龐大的 frames
        ).sort("created_at", DESCENDING)
        docs = list(cursor)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"無法讀取資料庫：{exc}")

    return [
        PoseSessionSummary(
            id=str(d["_id"]),
            label=d.get("label", ""),
            source_label=d.get("source_label", ""),
            frame_count=d.get("frame_count", 0),
            node_count=d.get("node_count", 0),
            total_steps=d.get("total_steps", 0),
            created_at=d.get("created_at", 0.0),
        )
        for d in docs
    ]


@app.get("/dataset/stats")
def dataset_stats(username: str = Depends(current_user)) -> dict:
    """目前資料集的統計：各標籤筆數，供確認訓練資料是否平衡。"""
    try:
        pipeline = [{"$group": {"_id": "$label", "count": {"$sum": 1}}}]
        agg = {row["_id"]: row["count"] for row in pose_sessions.aggregate(pipeline)}
        total = pose_sessions.count_documents({})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"無法讀取資料庫：{exc}")

    return {
        "total_sessions": total,
        "by_label": {"good": agg.get("good", 0), "bad": agg.get("bad", 0)},
    }


# ---------------------------------------------------------------------------
# 每幀即時串流上傳：開 session → 逐批 push frames → finish
# ---------------------------------------------------------------------------


class StartSessionIn(BaseModel):
    label: Optional[str] = None  # good / bad；預測模式可不帶（null）
    source_label: str = ""


class StartSessionResponse(BaseModel):
    session_id: str


class FramesIn(BaseModel):
    frames: List[PoseFrame]


class FinishIn(BaseModel):
    total_steps: int = 0
    left_steps: int = 0
    right_steps: int = 0
    avg_cadence_bpm: Optional[float] = None


def _parse_oid(session_id: str) -> ObjectId:
    try:
        return ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="session id 格式錯誤")


@app.post("/sessions", response_model=StartSessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(payload: StartSessionIn, username: str = Depends(current_user)) -> StartSessionResponse:
    label = payload.label.strip().lower() if payload.label else None
    if label is not None and label not in VALID_LABELS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="label 必須是 good 或 bad（或不帶）")

    doc = {
        "user": username,
        "label": label,
        "source_label": payload.source_label,
        "created_at": time.time(),
        "ended_at": None,
        "streaming": True,
        "frames": [],
        "frame_count": 0,
        "node_count": 0,
        "total_steps": 0,
        "left_steps": 0,
        "right_steps": 0,
        "avg_cadence_bpm": None,
    }
    try:
        result = pose_sessions.insert_one(doc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"無法建立 session：{exc}")
    return StartSessionResponse(session_id=str(result.inserted_id))


@app.post("/sessions/{session_id}/frames")
def append_frames(session_id: str, payload: FramesIn, username: str = Depends(current_user)) -> dict:
    if not payload.frames:
        return {"appended": 0}
    oid = _parse_oid(session_id)
    frame_docs = [f.model_dump() for f in payload.frames]
    node_count = sum(len(f.nodes) for f in payload.frames)
    try:
        result = pose_sessions.update_one(
            {"_id": oid, "user": username},
            {
                "$push": {"frames": {"$each": frame_docs}},
                "$inc": {"frame_count": len(frame_docs), "node_count": node_count},
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"無法寫入 frames：{exc}")
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="找不到 session")
    return {"appended": len(frame_docs)}


@app.post("/sessions/{session_id}/finish")
def finish_session(session_id: str, payload: FinishIn, username: str = Depends(current_user)) -> dict:
    oid = _parse_oid(session_id)
    try:
        result = pose_sessions.update_one(
            {"_id": oid, "user": username},
            {
                "$set": {
                    "ended_at": time.time(),
                    "streaming": False,
                    "total_steps": payload.total_steps,
                    "left_steps": payload.left_steps,
                    "right_steps": payload.right_steps,
                    "avg_cadence_bpm": payload.avg_cadence_bpm,
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"無法結束 session：{exc}")
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="找不到 session")
    return {"ok": True}


# ---------------------------------------------------------------------------
# 即時預測：載入訓練好的模型，回傳「好 / 壞」
# ---------------------------------------------------------------------------

_model_cache: Optional[dict] = None


def get_model() -> dict:
    global _model_cache
    if _model_cache is None:
        import joblib  # 延遲載入，避免無模型時也要求 sklearn 環境

        from train import MODEL_PATH

        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="尚未訓練模型，請先用 train.py 訓練並產生 pose_quality_model.joblib",
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


class PredictIn(BaseModel):
    frames: List[PoseFrame]


class PredictResponse(BaseModel):
    label: str
    probability_good: float
    probability_bad: float


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictIn, username: str = Depends(current_user)) -> PredictResponse:
    import numpy as np

    from train import session_features

    bundle = get_model()
    model = bundle["model"]

    feats = session_features({"frames": [f.model_dump() for f in payload.frames]})
    if feats is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="frames 為空，無法預測")

    proba = model.predict_proba(np.array([feats], dtype=float))[0]
    classes = list(model.classes_)
    p_good = float(proba[classes.index(1)]) if 1 in classes else 0.0
    p_bad = float(proba[classes.index(0)]) if 0 in classes else 0.0
    return PredictResponse(
        label="good" if p_good >= p_bad else "bad",
        probability_good=p_good,
        probability_bad=p_bad,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("POSE_RELOAD", "1") == "1"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
