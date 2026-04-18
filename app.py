import os
import logging
import tempfile
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings
from deepface import DeepFace
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

from spoof_detector import detect_spoof


# =========================
# CONFIG
# =========================

class Settings(BaseSettings):
    embed_dir: str = "embeddings"        # Dipakai ChromaDB sebagai persist_directory
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
# VECTOR STORE (ChromaDB)
# =========================

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None

COLLECTION_NAME = "face_embeddings"


def get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection (singleton)."""
    if _collection is None:
        raise RuntimeError("Vector store not initialized. Call init_vector_store() first.")
    return _collection


def init_vector_store(persist_directory: str) -> int:
    """
    Initialize ChromaDB dengan persistent storage.
    Mengembalikan jumlah embedding yang sudah tersimpan.
    """
    global _chroma_client, _collection

    os.makedirs(persist_directory, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # cosine distance → similarity = 1 - distance
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    count = _collection.count()
    logger.info(f"ChromaDB initialized — {count} embeddings loaded from '{persist_directory}'")
    return count


# ── CRUD helpers ─────────────────────────────────────────────

def load_embedding(user_id: str) -> Optional[list[float]]:
    """Ambil embedding berdasarkan user_id. Return None jika tidak ditemukan."""
    col = get_collection()
    result = col.get(ids=[user_id], include=["embeddings"])

    if not result["ids"]:
        return None

    return result["embeddings"][0]


def save_embedding(user_id: str, embedding: list[float]) -> None:
    """Simpan atau update embedding untuk user_id."""
    col = get_collection()

    # upsert: insert jika baru, update jika sudah ada
    col.upsert(
        ids=[user_id],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id}],
    )


def delete_embedding(user_id: str) -> bool:
    """Hapus embedding. Return True jika berhasil, False jika user tidak ada."""
    col = get_collection()

    result = col.get(ids=[user_id])
    if not result["ids"]:
        return False

    col.delete(ids=[user_id])
    return True


def query_similar(embedding: list[float], top_k: int = 1) -> list[dict]:
    """
    Opsional: cari embedding paling mirip di seluruh collection.
    Berguna untuk identifikasi (1:N matching).
    Mengembalikan list of {"user_id", "similarity", "distance"}.
    """
    col = get_collection()

    if col.count() == 0:
        return []

    results = col.query(
        query_embeddings=[embedding],
        n_results=min(top_k, col.count()),
        include=["metadatas", "distances"],
    )

    output = []
    for uid, dist in zip(results["ids"][0], results["distances"][0]):
        # ChromaDB cosine: distance = 1 - cosine_similarity
        similarity = round(1.0 - dist, 4)
        output.append({"user_id": uid, "similarity": similarity, "distance": round(dist, 4)})

    return output


# =========================
# LIFESPAN
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    count = init_vector_store(cfg.embed_dir)
    logger.info(f"Face API ready — {count} users registered")

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
    anti_spoofing: bool = True
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
            detector_backend=cfg.detector_backend,
        )
        return results[0]["embedding"]
    except Exception as e:
        raise ValueError(f"Face not detected: {e}") from e


# =========================
# ROUTES
# =========================

@app.post("/register")
async def register(data: FaceRequest):
    cfg = get_settings()
    temp_path = None

    try:
        temp_path = await download_image(data.image_url)

        if data.anti_spoofing:
            spoof_result = detect_spoof(temp_path)
            if not spoof_result.is_real:
                resp = {"status": False, "message": spoof_result.reason or "Spoof detected"}
                if data.debug:
                    resp["spoof_scores"] = spoof_result.scores
                return resp

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

        if data.anti_spoofing:
            spoof_result = detect_spoof(temp_path)
            if not spoof_result.is_real:
                resp = {"status": False, "message": spoof_result.reason or "Spoof detected"}
                if data.debug:
                    resp["spoof_scores"] = spoof_result.scores
                return resp

        try:
            new_embedding = extract_embedding(temp_path, cfg)
        except ValueError as e:
            return {"status": False, "message": str(e)}

        # ── Cosine similarity via ChromaDB query ─────────────────
        # Query top-1 untuk user_id ini secara langsung dari stored_embedding
        matches = query_similar(new_embedding, top_k=1)

        # Filter hasil hanya untuk user_id yang diminta
        similarity = 0.0
        for match in matches:
            if match["user_id"] == data.user_id:
                similarity = match["similarity"]
                break

        # Fallback: hitung manual jika query tidak return user ini
        # (terjadi bila ada lebih banyak user & top_k terlalu kecil)
        if similarity == 0.0:
            import numpy as np
            va = np.array(stored_embedding)
            vb = np.array(new_embedding)
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            similarity = float(np.dot(va, vb) / denom) if denom else 0.0

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
    col = get_collection()
    registered = col.count()
    cfg = get_settings()
    return {
        "status": "ok",
        "registered_users": registered,
        "vector_store": "chromadb",
        "model": cfg.face_model,
        "detector": cfg.detector_backend,
    }


# ── Endpoint baru: identifikasi 1:N (bonus) ──────────────────

@app.post("/identify")
async def identify(data: FaceRequest, top_k: int = 5):
    """
    Cari siapa pemilik wajah dari seluruh database (1:N matching).
    Mengembalikan top-K kandidat beserta similarity score.
    """
    cfg = get_settings()
    temp_path = None

    try:
        temp_path = await download_image(data.image_url)

        if data.anti_spoofing:
            spoof_result = detect_spoof(temp_path)
            if not spoof_result.is_real:
                resp = {"status": False, "message": spoof_result.reason or "Spoof detected"}
                if data.debug:
                    resp["spoof_scores"] = spoof_result.scores
                return resp

        try:
            embedding = extract_embedding(temp_path, cfg)
        except ValueError as e:
            return {"status": False, "message": str(e)}

        candidates = query_similar(embedding, top_k=top_k)

        return {
            "status": True,
            "candidates": candidates,   # [{user_id, similarity, distance}, ...]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during identify")
        return {"status": False, "message": str(e)}
    finally:
        safe_remove(temp_path)