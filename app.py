import os
import pickle
import logging
import tempfile
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import httpx
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

from spoof_detector import detect_spoof


# =========================
# CONFIG
# =========================

class Settings(BaseSettings):
    embed_dir: str = "embeddings"
    face_model: str = "Facenet"
    detector_backend: str = "opencv"
    verify_threshold: float = 0.6
    download_timeout: int = 10
    max_image_bytes: int = 5 * 1024 * 1024  # 5 MB

    class Config:
        env_prefix = "FACEAPI_"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("face_api")


# =========================
# IN-MEMORY CACHE
# =========================

_embedding_cache: dict[str, list[float]] = {}


def load_embedding(user_id: str) -> Optional[list[float]]:
    if user_id in _embedding_cache:
        return _embedding_cache[user_id]

    cfg = get_settings()
    path = os.path.join(cfg.embed_dir, f"{user_id}.pkl")

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        embedding = pickle.load(f)

    _embedding_cache[user_id] = embedding
    return embedding


def save_embedding(user_id: str, embedding: list[float]) -> None:
    cfg = get_settings()
    path = os.path.join(cfg.embed_dir, f"{user_id}.pkl")

    with open(path, "wb") as f:
        pickle.dump(embedding, f)

    _embedding_cache[user_id] = embedding


def delete_embedding(user_id: str) -> bool:
    cfg = get_settings()
    path = os.path.join(cfg.embed_dir, f"{user_id}.pkl")

    _embedding_cache.pop(user_id, None)

    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# =========================
# LIFESPAN
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    os.makedirs(cfg.embed_dir, exist_ok=True)

    loaded = 0
    for fname in os.listdir(cfg.embed_dir):
        if fname.endswith(".pkl"):
            user_id = fname[:-4]
            if load_embedding(user_id):
                loaded += 1
    logger.info(f"Pre-loaded {loaded} embeddings into cache")

    yield

    logger.info("Shutting down face API")


# =========================
# APP
# =========================

app = FastAPI(title="Face Recognition API", lifespan=lifespan)


# =========================
# SCHEMAS
# =========================

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


class FaceRequest(BaseModel):
    user_id: str
    image_url: str
    # anti_spoofing sekarang selalu True secara default
    anti_spoofing: bool = True
    # expose raw scores untuk debugging (jangan aktifkan di production)
    debug: bool = False

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not _SAFE_ID.match(v):
            raise ValueError("user_id must be 1-64 alphanumeric/underscore/dash characters")
        return v

    @field_validator("image_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("image_url must start with http:// or https://")
        return v


class VerifyRequest(FaceRequest):
    threshold: Optional[float] = None

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError("threshold must be between 0 and 1")
        return v


# =========================
# HELPERS
# =========================

async def download_image(url: str) -> str:
    cfg = get_settings()

    async with httpx.AsyncClient(timeout=cfg.download_timeout) as client:
        response = await client.get(url)
        response.raise_for_status()

        data = response.content
        if len(data) > cfg.max_image_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image too large"
            )

    suffix = ".jpg"
    content_type = response.headers.get("content-type", "")
    if "png" in content_type:
        suffix = ".png"
    elif "webp" in content_type:
        suffix = ".webp"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()

    return tmp.name


def safe_remove(path: Optional[str]) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning(f"Could not remove temp file {path}: {e}")


def extract_embedding(img_path: str, cfg: Settings) -> list[float]:
    try:
        results = DeepFace.represent(
            img_path=img_path,
            model_name=cfg.face_model,
            enforce_detection=True,
            detector_backend=cfg.detector_backend
        )
        return results[0]["embedding"]
    except Exception as e:
        raise ValueError(f"Face not detected: {e}") from e


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# =========================
# ROUTES
# =========================

@app.post("/register")
async def register(data: FaceRequest):
    cfg = get_settings()
    temp_path = None

    try:
        temp_path = await download_image(data.image_url)

        # ── Anti-spoofing (multi-layer) ──────────────────────────
        if data.anti_spoofing:
            spoof_result = detect_spoof(temp_path)
            if not spoof_result.is_real:
                resp = {
                    "status": False,
                    "message": spoof_result.reason or "Spoof detected",
                }
                if data.debug:
                    resp["spoof_scores"] = spoof_result.scores
                return resp

        # ── Extract & save embedding ─────────────────────────────
        try:
            embedding = extract_embedding(temp_path, cfg)
        except ValueError as e:
            return {"status": False, "message": str(e)}

        save_embedding(data.user_id, embedding)
        logger.info(f"Registered face for user: {data.user_id}")

        return {"status": True, "message": "Face registered successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during register for {data.user_id}")
        return {"status": False, "message": str(e)}
    finally:
        safe_remove(temp_path)


@app.post("/verify")
async def verify(data: VerifyRequest):
    cfg = get_settings()
    threshold = data.threshold if data.threshold is not None else cfg.verify_threshold
    temp_path = None

    stored_embedding = load_embedding(data.user_id)
    if stored_embedding is None:
        return {"status": False, "message": "User not registered"}

    try:
        temp_path = await download_image(data.image_url)

        # ── Anti-spoofing (multi-layer) ──────────────────────────
        if data.anti_spoofing:
            spoof_result = detect_spoof(temp_path)
            if not spoof_result.is_real:
                resp = {
                    "status": False,
                    "message": spoof_result.reason or "Spoof detected",
                }
                if data.debug:
                    resp["spoof_scores"] = spoof_result.scores
                return resp

        # ── Extract & compare embedding ──────────────────────────
        try:
            new_embedding = extract_embedding(temp_path, cfg)
        except ValueError as e:
            return {"status": False, "message": str(e)}

        similarity = cosine_similarity(stored_embedding, new_embedding)
        verified = similarity >= threshold

        logger.info(
            f"Verify user={data.user_id} similarity={similarity:.4f} "
            f"threshold={threshold} verified={verified}"
        )

        return {
            "status": True,
            "verified": verified,
            "similarity": round(similarity, 4),
            "threshold": threshold,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during verify for {data.user_id}")
        return {"status": False, "message": str(e)}
    finally:
        safe_remove(temp_path)


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    if not _SAFE_ID.match(user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")

    deleted = delete_embedding(user_id)
    if not deleted:
        return {"status": False, "message": "User not found"}

    logger.info(f"Deleted embedding for user: {user_id}")
    return {"status": True, "message": "User deleted"}


@app.get("/health")
async def health():
    cfg = get_settings()
    registered = len([f for f in os.listdir(cfg.embed_dir) if f.endswith(".pkl")])
    return {
        "status": "ok",
        "registered_users": registered,
        "cached_users": len(_embedding_cache),
        "model": cfg.face_model,
        "detector": cfg.detector_backend,
    }
