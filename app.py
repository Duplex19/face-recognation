"""
app.py
FastAPI Face Recognition + Attendance System
Optimized for small teams (≤ 20 users, ~16 requests/day)

Endpoints:
  POST   /register          — daftarkan wajah pegawai
  POST   /verify            — verifikasi wajah saja
  POST   /absen             — absen masuk / pulang (verify + log)
  GET    /rekap             — rekap absensi (filter by date / user)
  GET    /rekap/{user_id}   — rekap per pegawai
  DELETE /users/{user_id}   — hapus data wajah pegawai
  GET    /health            — status sistem

Optimizations:
  - Model DeepFace di-preload saat startup (eliminasi cold-start)
  - Semaphore membatasi maks. 2 inference bersamaan (cegah OOM)
  - Absensi tersimpan di SQLite (ringan, cukup untuk 8 pegawai)
  - Validasi duplikasi absen (tidak bisa absen masuk 2x sehari)
"""

import asyncio
import logging
import os
import pickle
import re
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime
from functools import lru_cache
from typing import Optional

import cv2
import httpx
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

from spoof_detector import detect_spoof


# =============================================================================
# CONFIG
# =============================================================================

class Settings(BaseSettings):
    embed_dir: str = "embeddings"
    db_path: str = "attendance.db"
    face_model: str = "Facenet"
    detector_backend: str = "opencv"
    verify_threshold: float = 0.6
    download_timeout: int = 10
    max_image_bytes: int = 5 * 1024 * 1024   # 5 MB
    # Maks. concurrent spoof+inference agar RAM tidak meledak
    max_concurrent_inference: int = 2

    class Config:
        env_prefix = "FACEAPI_"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("face_api")


# =============================================================================
# SEMAPHORE  (dibuat setelah event loop ready — lihat lifespan)
# =============================================================================

_inference_semaphore: asyncio.Semaphore = None   # type: ignore


# =============================================================================
# IN-MEMORY EMBEDDING CACHE
# =============================================================================

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


# =============================================================================
# SQLITE — ATTENDANCE
# =============================================================================

def get_db_conn() -> sqlite3.Connection:
    cfg = get_settings()
    conn = sqlite3.connect(cfg.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Buat tabel attendance jika belum ada."""
    conn = get_db_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT    NOT NULL,
            tipe      TEXT    NOT NULL CHECK(tipe IN ('masuk', 'pulang')),
            timestamp TEXT    NOT NULL,
            tanggal   TEXT    NOT NULL,
            similarity REAL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database attendance siap.")


def log_attendance(user_id: str, tipe: str, similarity: float) -> None:
    now = datetime.now()
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO attendance (user_id, tipe, timestamp, tanggal, similarity)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, tipe, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"), similarity),
    )
    conn.commit()
    conn.close()


def already_absen(user_id: str, tipe: str, tanggal: str) -> bool:
    """Cek apakah user sudah absen tipe tsb hari ini."""
    conn = get_db_conn()
    row = conn.execute(
        "SELECT 1 FROM attendance WHERE user_id=? AND tipe=? AND tanggal=?",
        (user_id, tipe, tanggal),
    ).fetchone()
    conn.close()
    return row is not None


def get_rekap(
    user_id: Optional[str] = None,
    tanggal_mulai: Optional[str] = None,
    tanggal_selesai: Optional[str] = None,
) -> list[dict]:
    conn = get_db_conn()
    query = "SELECT * FROM attendance WHERE 1=1"
    params: list = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    if tanggal_mulai:
        query += " AND tanggal >= ?"
        params.append(tanggal_mulai)
    if tanggal_selesai:
        query += " AND tanggal <= ?"
        params.append(tanggal_selesai)

    query += " ORDER BY timestamp DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =============================================================================
# LIFESPAN — startup & shutdown
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_semaphore

    cfg = get_settings()

    # Buat folder embeddings
    os.makedirs(cfg.embed_dir, exist_ok=True)

    # Inisialisasi database
    init_db()

    # Semaphore untuk batasi concurrent inference
    _inference_semaphore = asyncio.Semaphore(cfg.max_concurrent_inference)

    # Pre-load semua embedding yang sudah ada
    loaded = 0
    for fname in os.listdir(cfg.embed_dir):
        if fname.endswith(".pkl"):
            user_id = fname[:-4]
            if load_embedding(user_id):
                loaded += 1
    logger.info(f"Pre-loaded {loaded} embedding(s) into cache.")

    # Preload DeepFace model — eliminasi cold-start saat request pertama
    logger.info("Preloading DeepFace model, harap tunggu...")
    try:
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, dummy)
            tmp_path = tmp.name
        try:
            DeepFace.represent(
                img_path=tmp_path,
                model_name=cfg.face_model,
                enforce_detection=False,
                detector_backend=cfg.detector_backend,
            )
        except Exception:
            pass
        finally:
            _safe_remove(tmp_path)
    except Exception as e:
        logger.warning(f"Model preload warning (tidak kritis): {e}")

    logger.info("✅ Server siap menerima request.")
    yield
    logger.info("Server shutdown.")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="Face Attendance API", version="2.0.0", lifespan=lifespan)


# =============================================================================
# SCHEMAS
# =============================================================================

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
            raise ValueError("user_id: hanya huruf, angka, underscore, dash. Maks 64 karakter.")
        return v

    @field_validator("image_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("image_url harus diawali http:// atau https://")
        return v


class VerifyRequest(FaceRequest):
    threshold: Optional[float] = None

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError("threshold harus antara 0 dan 1")
        return v


class AbsenRequest(VerifyRequest):
    tipe: str   # "masuk" atau "pulang"

    @field_validator("tipe")
    @classmethod
    def validate_tipe(cls, v: str) -> str:
        if v not in ("masuk", "pulang"):
            raise ValueError("tipe harus 'masuk' atau 'pulang'")
        return v


# =============================================================================
# HELPERS
# =============================================================================

async def download_image(url: str) -> str:
    """Download gambar dari URL, simpan ke file temporary, return path-nya."""
    cfg = get_settings()
    async with httpx.AsyncClient(timeout=cfg.download_timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.content
        if len(data) > cfg.max_image_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Ukuran gambar melebihi batas {cfg.max_image_bytes // 1024 // 1024} MB.",
            )

    content_type = response.headers.get("content-type", "")
    suffix = ".png" if "png" in content_type else ".webp" if "webp" in content_type else ".jpg"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _safe_remove(path: Optional[str]) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning(f"Gagal hapus file temp {path}: {e}")


def extract_embedding(img_path: str, cfg: Settings) -> list[float]:
    results = DeepFace.represent(
        img_path=img_path,
        model_name=cfg.face_model,
        enforce_detection=True,
        detector_backend=cfg.detector_backend,
    )
    return results[0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


async def run_spoof_and_embed(temp_path: str, cfg: Settings, anti_spoofing: bool):
    """
    Jalankan spoof detection + embedding extraction dengan semaphore
    agar tidak lebih dari N request berjalan bersamaan.
    Returns (spoof_result_or_None, embedding_or_None, error_str_or_None)
    """
    loop = asyncio.get_event_loop()

    async with _inference_semaphore:
        # Spoof detection (CPU-bound → jalankan di thread pool)
        if anti_spoofing:
            spoof_result = await loop.run_in_executor(None, detect_spoof, temp_path)
            if not spoof_result.is_real:
                return spoof_result, None, None

        # Face embedding (CPU-bound)
        try:
            embedding = await loop.run_in_executor(None, extract_embedding, temp_path, cfg)
        except Exception as e:
            return None, None, f"Wajah tidak terdeteksi: {e}"

    return None, embedding, None


# =============================================================================
# ROUTES
# =============================================================================

@app.post("/register", summary="Daftarkan wajah pegawai")
async def register(data: FaceRequest):
    """
    Daftarkan wajah pegawai baru.
    Jika user_id sudah ada, embedding-nya akan diperbarui.
    """
    cfg = get_settings()
    temp_path = None

    try:
        temp_path = await download_image(data.image_url)
        spoof_result, embedding, error = await run_spoof_and_embed(
            temp_path, cfg, data.anti_spoofing
        )

        if spoof_result and not spoof_result.is_real:
            resp = {"status": False, "message": spoof_result.reason or "Spoof terdeteksi"}
            if data.debug:
                resp["spoof_scores"] = spoof_result.scores
            return resp

        if error:
            return {"status": False, "message": error}

        save_embedding(data.user_id, embedding)
        logger.info(f"Registered: {data.user_id}")
        return {"status": True, "message": f"Wajah '{data.user_id}' berhasil didaftarkan."}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error register {data.user_id}")
        return {"status": False, "message": str(e)}
    finally:
        _safe_remove(temp_path)


@app.post("/verify", summary="Verifikasi wajah")
async def verify(data: VerifyRequest):
    """
    Verifikasi apakah wajah cocok dengan data terdaftar.
    Tidak mencatat absensi — gunakan /absen untuk itu.
    """
    cfg = get_settings()
    threshold = data.threshold if data.threshold is not None else cfg.verify_threshold
    temp_path = None

    stored_embedding = load_embedding(data.user_id)
    if stored_embedding is None:
        return {"status": False, "message": f"User '{data.user_id}' belum terdaftar."}

    try:
        temp_path = await download_image(data.image_url)
        spoof_result, new_embedding, error = await run_spoof_and_embed(
            temp_path, cfg, data.anti_spoofing
        )

        if spoof_result and not spoof_result.is_real:
            resp = {"status": False, "message": spoof_result.reason or "Spoof terdeteksi"}
            if data.debug:
                resp["spoof_scores"] = spoof_result.scores
            return resp

        if error:
            return {"status": False, "message": error}

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
        logger.exception(f"Error verify {data.user_id}")
        return {"status": False, "message": str(e)}
    finally:
        _safe_remove(temp_path)


@app.post("/absen", summary="Absen masuk atau pulang")
async def absen(data: AbsenRequest):
    """
    Absen masuk atau pulang untuk pegawai.
    - Validasi wajah (verify) dilakukan terlebih dahulu.
    - Satu pegawai hanya bisa absen masuk 1x dan pulang 1x per hari.
    - Hasil dicatat di database SQLite.
    """
    cfg = get_settings()
    threshold = data.threshold if data.threshold is not None else cfg.verify_threshold
    temp_path = None

    stored_embedding = load_embedding(data.user_id)
    if stored_embedding is None:
        return {"status": False, "message": f"User '{data.user_id}' belum terdaftar."}

    # Cek duplikasi absen hari ini
    today = date.today().strftime("%Y-%m-%d")
    if already_absen(data.user_id, data.tipe, today):
        return {
            "status": False,
            "message": f"anda sudah absen {data.tipe} hari ini ({today}).",
        }

    try:
        temp_path = await download_image(data.image_url)
        spoof_result, new_embedding, error = await run_spoof_and_embed(
            temp_path, cfg, data.anti_spoofing
        )

        if spoof_result and not spoof_result.is_real:
            resp = {
                "status": False,
                "message": f"Absen ditolak — {spoof_result.reason or 'Spoof terdeteksi'}",
            }
            if data.debug:
                resp["spoof_scores"] = spoof_result.scores
            return resp

        if error:
            return {"status": False, "message": error}

        similarity = cosine_similarity(stored_embedding, new_embedding)
        verified = similarity >= threshold

        if not verified:
            logger.info(
                f"Absen GAGAL user={data.user_id} tipe={data.tipe} "
                f"similarity={similarity:.4f} threshold={threshold}"
            )
            return {
                "status": False,
                "message": "Wajah tidak dikenali. Silakan coba lagi.",
                "similarity": round(similarity, 4),
            }

        # Catat absensi
        log_attendance(data.user_id, data.tipe, round(similarity, 4))
        waktu_sekarang = datetime.now().strftime("%H:%M:%S")

        logger.info(
            f"Absen BERHASIL user={data.user_id} tipe={data.tipe} "
            f"similarity={similarity:.4f} jam={waktu_sekarang}"
        )

        return {
            "status": True,
            "message": f"Absen {data.tipe} berhasil!",
            "user_id": data.user_id,
            "tipe": data.tipe,
            "tanggal": today,
            "jam": waktu_sekarang,
            "similarity": round(similarity, 4),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error absen {data.user_id}")
        return {"status": False, "message": str(e)}
    finally:
        _safe_remove(temp_path)


@app.get("/rekap", summary="Rekap absensi semua pegawai")
async def rekap(
    user_id: Optional[str] = None,
    tanggal_mulai: Optional[str] = None,
    tanggal_selesai: Optional[str] = None,
):
    """
    Ambil rekap absensi.
    Query params (opsional):
      - user_id        : filter per pegawai
      - tanggal_mulai  : format YYYY-MM-DD
      - tanggal_selesai: format YYYY-MM-DD
    """
    data = get_rekap(user_id, tanggal_mulai, tanggal_selesai)
    return {"status": True, "total": len(data), "data": data}


@app.get("/rekap/{user_id}", summary="Rekap absensi per pegawai")
async def rekap_user(
    user_id: str,
    tanggal_mulai: Optional[str] = None,
    tanggal_selesai: Optional[str] = None,
):
    """Rekap absensi untuk satu pegawai tertentu."""
    if not _SAFE_ID.match(user_id):
        raise HTTPException(status_code=400, detail="user_id tidak valid.")
    data = get_rekap(user_id, tanggal_mulai, tanggal_selesai)
    return {"status": True, "user_id": user_id, "total": len(data), "data": data}


@app.delete("/users/{user_id}", summary="Hapus data wajah pegawai")
async def delete_user(user_id: str):
    if not _SAFE_ID.match(user_id):
        raise HTTPException(status_code=400, detail="user_id tidak valid.")
    deleted = delete_embedding(user_id)
    if not deleted:
        return {"status": False, "message": f"User '{user_id}' tidak ditemukan."}
    logger.info(f"Deleted embedding: {user_id}")
    return {"status": True, "message": f"Data wajah '{user_id}' berhasil dihapus."}


@app.get("/health", summary="Status server")
async def health():
    cfg = get_settings()
    registered = len([f for f in os.listdir(cfg.embed_dir) if f.endswith(".pkl")])
    today = date.today().strftime("%Y-%m-%d")
    absen_hari_ini = get_rekap(tanggal_mulai=today, tanggal_selesai=today)
    return {
        "status": "ok",
        "registered_users": registered,
        "cached_users": len(_embedding_cache),
        "absen_hari_ini": len(absen_hari_ini),
        "model": cfg.face_model,
        "detector": cfg.detector_backend,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }