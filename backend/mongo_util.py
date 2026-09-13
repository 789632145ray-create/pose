"""MongoDB 連線（本機 / Atlas / Railway 容器共用）。"""

from __future__ import annotations

import os

import certifi
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError


def make_mongo_client(url: str | None = None, timeout_ms: int = 15_000) -> MongoClient:
    mongo_url = (url or os.environ.get("POSE_MONGO_URL", "mongodb://localhost:27017")).strip()
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
