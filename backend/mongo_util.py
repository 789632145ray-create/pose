"""MongoDB 連線（本機 / Atlas / Railway 容器共用）。"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import certifi
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError


def describe_mongo_url_problem(url: str) -> str | None:
    """檢查連線字串是否明顯不可用（文件範例、空主機、空 DNS 標籤）。"""
    raw = (url or "").strip().strip('"').strip("'")
    if not raw:
        return "POSE_MONGO_URL 是空的。"
    lowered = raw.lower()
    if lowered in {"mongodb+srv://", "mongodb+srv://...", "mongodb://", "mongodb://..."}:
        return "這是文件裡的範例字串，不是真正的 Atlas 連線。"
    if "..." in raw or "xxxxx" in lowered or "你的" in raw or "帳號:密碼" in raw:
        return "連線字串還是範例（含 ... / xxxxx），請到 Railway → Variables 複製完整的 POSE_MONGO_URL。"
    parsed = urlparse(raw)
    host = (parsed.hostname or "").strip()
    if not host:
        return "連線字串沒有主機名稱（@ 後面是空的）。密碼若含 @ # % 必須先做 URL 編碼（@ 改成 %40）。"
    if host.startswith(".") or host.endswith(".") or ".." in host:
        return f"主機名稱有空的 DNS 標籤（{host}）。常見原因是複製了 mongodb+srv://... 而不是完整 cluster 網址。"
    return None


def mongo_url_help(url: str, exc: BaseException | None = None) -> str:
    reason = describe_mongo_url_problem(url) or (str(exc) if exc else "連線字串無法解析")
    return "\n".join(
        [
            f"MongoDB 連線字串無效：{reason}",
            "",
            "不要複製文件裡的 mongodb+srv://...",
            "請到 Railway 專案 → Variables，把 POSE_MONGO_URL 整段複製下來，長得像：",
            "  mongodb+srv://使用者:密碼@cluster0.abcd123.mongodb.net/?retryWrites=true&w=majority",
            "",
            "或改從手機匯出 JSON，不需要 Atlas 連線：",
            "  python train.py --engine mediapipe --from-json 匯出的檔案.json",
        ]
    )


def make_mongo_client(url: str | None = None, timeout_ms: int = 15_000) -> MongoClient:
    mongo_url = (url or os.environ.get("POSE_MONGO_URL", "mongodb://localhost:27017")).strip()
    problem = describe_mongo_url_problem(mongo_url)
    if problem:
        raise ConfigurationError(mongo_url_help(mongo_url))
    kwargs: dict = {
        "serverSelectionTimeoutMS": timeout_ms,
        "connectTimeoutMS": timeout_ms,
        "socketTimeoutMS": 60_000,
    }
    use_tls = mongo_url.startswith("mongodb+srv://") or "tls=true" in mongo_url.lower() or "ssl=true" in mongo_url.lower()
    if use_tls:
        kwargs["tls"] = True
        kwargs["tlsCAFile"] = certifi.where()
        # Docker / Railway 容器常因 OCSP 檢查失敗導致 SSL handshake failed。
        kwargs["tlsDisableOCSPEndpointCheck"] = True
    return MongoClient(mongo_url, **kwargs)


def ping_mongo(client: MongoClient | None = None) -> tuple[bool, str]:
    """回傳 (成功與否, 訊息)。"""
    own = client is None
    c = client or make_mongo_client()
    try:
        c.admin.command("ping")
        return True, "mongodb ok"
    except (ServerSelectionTimeoutError, ConfigurationError) as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        if own:
            c.close()
