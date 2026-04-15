"""
spoof_detector.py
Multi-layer presentation attack detection.
Detects: screen display (monitor/phone), printed photos.

Layers:
  1. DeepFace liveness (MiniFASNet)
  2. Moiré / screen-pixel pattern via FFT
  3. LBP texture uniformity (print attack)
  4. HSV backlight signature (screen glow)
  5. Laplacian gradient richness (blur / flat print)
"""

import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("spoof_detector")


# ─────────────────────────────────────────
# Result type
# ─────────────────────────────────────────

@dataclass
class SpoofResult:
    is_real: bool
    reason: Optional[str] = None          # filled when is_real=False
    scores: dict = None                   # debug: individual layer scores

    def __post_init__(self):
        if self.scores is None:
            self.scores = {}


# ─────────────────────────────────────────
# Config / thresholds  (tunable via env)
# ─────────────────────────────────────────

from pydantic_settings import BaseSettings
from functools import lru_cache

# class SpoofConfig(BaseSettings):
#     # FFT — ratio of high-freq energy peaks expected for screens
#     fft_peak_ratio_threshold: float = 0.018

#     # LBP — uniformity score; higher = more uniform = more likely print
#     lbp_uniformity_threshold: float = 0.55

#     # HSV — screen backlight tends to push high V channel mean
#     hsv_v_mean_threshold: float = 210.0

#     # Laplacian variance — real faces have rich gradients
#     laplacian_var_threshold: float = 80.0

#     # How many layers must flag as spoof to reject
#     # (1 = strict, 2-3 = balanced)
#     min_spoof_votes: int = 2

#     class Config:
#         env_prefix = "SPOOF_"

class SpoofConfig(BaseSettings):
    # FFT — Naikkan sedikit agar tidak sensitif terhadap pola latar belakang
    fft_peak_ratio_threshold: float = 0.023  # Sebelumnya 0.018
    
    # LBP — Naikkan toleransi untuk wajah yang memakai makeup tebal / kulit sangat halus
    lbp_uniformity_threshold: float = 0.55 #0.60   # Sebelumnya 0.55
    
    # HSV — Hanya tangkap layar yang benar-benar "over-exposed" / bercahaya aneh
    hsv_v_mean_threshold: float = 235.0      # Sebelumnya 210.0
    
    # Laplacian variance — Toleransi untuk kamera webcam/HP yang kurang fokus
    laplacian_var_threshold: float = 50.0    # Sebelumnya 80.0
    laplacian_var_max_threshold: float = 88.0    # Sebelumnya 80.0
    
    # Minimum vote tetap 2, ini sudah merupakan batas ideal (balance)
    min_spoof_votes: int = 2

    class Config:
        env_prefix = "SPOOF_"
        
@lru_cache
def get_spoof_config() -> SpoofConfig:
    return SpoofConfig()

# ─────────────────────────────────────────
# Helper: crop face ROI
# ─────────────────────────────────────────

def _crop_face_roi(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect face and return a tight crop.
    Falls back to center-crop if detection fails.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # fallback: center 60% crop
        h, w = img_bgr.shape[:2]
        m = 0.2
        return img_bgr[int(h * m): int(h * (1 - m)), int(w * m): int(w * (1 - m))]

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    # small padding
    pad = int(min(fw, fh) * 0.1)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + fw + pad)
    y2 = min(img_bgr.shape[0], y + fh + pad)
    return img_bgr[y1:y2, x1:x2]


# ─────────────────────────────────────────
# Layer 1 — DeepFace liveness
# ─────────────────────────────────────────

def _check_deepface_liveness(img_path: str) -> tuple[bool, float]:
    """
    Returns (is_real, confidence).
    Confidence is the antispoof score from DeepFace (0–1).
    """
    try:
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            img_path=img_path,
            anti_spoofing=True,
            enforce_detection=True,
        )
        if not faces:
            return False, 0.0
        f = faces[0]
        is_real = bool(f.get("is_real", False))
        score = float(f.get("antispoof_score", 0.5))
        return is_real, score
    except Exception as e:
        logger.warning(f"DeepFace liveness error: {e}")
        return False, 0.0


# ─────────────────────────────────────────
# Layer 2 — FFT Moiré / screen-pixel pattern
# ─────────────────────────────────────────

def _fft_screen_score(face_roi: np.ndarray) -> float:
    """
    Screens have a regular pixel-grid → peaks in FFT magnitude spectrum.
    Returns the ratio of dominant high-frequency peak energy to total energy.
    Higher score = more screen-like.
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256)).astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    # mask out DC (center 10%)
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * 0.05)
    magnitude[cy - r: cy + r, cx - r: cx + r] = 0

    total_energy = magnitude.sum() + 1e-9
    # find top-N peaks energy
    flat = magnitude.flatten()
    flat.sort()
    top_n = flat[-50:]          # top 50 pixels
    peak_energy = top_n.sum()

    score = peak_energy / total_energy
    return float(score)


# ─────────────────────────────────────────
# Layer 3 — LBP texture uniformity (print)
# ─────────────────────────────────────────

def _lbp_uniformity_score(face_roi: np.ndarray) -> float:
    """
    Local Binary Pattern uniformity.
    Real faces have diverse micro-texture.
    Printed/screen faces tend toward uniform LBP patterns.
    Returns uniformity score 0–1 (higher = more uniform = more likely spoof).
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    radius = 1
    n_points = 8

    lbp_img = np.zeros_like(gray, dtype=np.uint8)
    h, w = gray.shape

    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray[i, j]
            code = 0
            # 8 neighbours clockwise
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0,  1),
                (1,  1),  (1, 0),  (1, -1),
                (0, -1),
            ]
            for k, (dy, dx) in enumerate(offsets):
                if gray[i + dy, j + dx] >= center:
                    code |= (1 << k)
            lbp_img[i, j] = code

    hist, _ = np.histogram(lbp_img, bins=256, range=(0, 255))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-9

    # entropy-based uniformity: low entropy = high uniformity
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    max_entropy = np.log2(256)
    uniformity = 1.0 - (entropy / max_entropy)

    return float(uniformity)


# ─────────────────────────────────────────
# Layer 4 — HSV backlight / screen glow
# ─────────────────────────────────────────

def _hsv_screen_score(face_roi: np.ndarray) -> float:
    """
    Screen images have characteristically high V (brightness) channel
    from the backlight and unnaturally high S (saturation).
    Returns mean V channel value (0–255). Higher = more screen-like.
    """
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    v_mean = float(hsv[:, :, 2].mean())
    return v_mean


# ─────────────────────────────────────────
# Layer 5 — Laplacian gradient richness
# ─────────────────────────────────────────

def _laplacian_variance(face_roi: np.ndarray) -> float:
    """
    Real faces photographed directly have rich, sharp micro-gradients.
    Re-photographed images (screen/print) lose high-frequency detail.
    Returns Laplacian variance; lower = blurrier = more likely spoof.
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


# ─────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────

def detect_spoof(img_path: str, force_all_layers: bool = False) -> SpoofResult:
    """
    Run all spoof detection layers and return a combined result.

    Args:
        img_path: Path to the image file.
        force_all_layers: If True, run all CV layers even if DeepFace
                          already flagged spoof (useful for debugging).

    Returns:
        SpoofResult with is_real, reason, and per-layer scores.
    """
    cfg = get_spoof_config()
    scores: dict = {}
    spoof_votes: list[str] = []

    # ── Load image ──────────────────────────────────────────────
    img = cv2.imread(img_path)
    if img is None:
        return SpoofResult(is_real=False, reason="Could not read image", scores={})

    face_roi = _crop_face_roi(img)

    # ── Layer 1: DeepFace liveness ──────────────────────────────
    df_real, df_score = _check_deepface_liveness(img_path)
    scores["deepface_liveness"] = df_score
    scores["deepface_is_real"] = int(df_real)

    if not df_real:
        spoof_votes.append("deepface_liveness")
        if not force_all_layers and cfg.min_spoof_votes == 1:
            return SpoofResult(
                is_real=False,
                reason="Liveness check failed",
                scores=scores
            )

    # ── Layer 2: FFT screen pattern ─────────────────────────────
    fft_score = _fft_screen_score(face_roi)
    scores["fft_peak_ratio"] = round(fft_score, 5)
    if fft_score > cfg.fft_peak_ratio_threshold:
        spoof_votes.append("fft_screen_pattern")
        logger.debug(f"FFT screen flag: {fft_score:.5f} > {cfg.fft_peak_ratio_threshold}")

    # ── Layer 3: LBP texture ────────────────────────────────────
    lbp_score = _lbp_uniformity_score(face_roi)
    scores["lbp_uniformity"] = round(lbp_score, 4)
    if lbp_score > cfg.lbp_uniformity_threshold:
        spoof_votes.append("lbp_texture_uniform")
        logger.debug(f"LBP uniform flag: {lbp_score:.4f} > {cfg.lbp_uniformity_threshold}")

    # ── Layer 4: HSV brightness ─────────────────────────────────
    hsv_score = _hsv_screen_score(face_roi)
    scores["hsv_v_mean"] = round(hsv_score, 2)
    if hsv_score > cfg.hsv_v_mean_threshold:
        spoof_votes.append("hsv_backlight")
        logger.debug(f"HSV backlight flag: {hsv_score:.2f} > {cfg.hsv_v_mean_threshold}")

    # ── Layer 5: Laplacian sharpness ────────────────────────────
    lap_var = _laplacian_variance(face_roi)
    scores["laplacian_variance"] = round(lap_var, 2)
    if lap_var < cfg.laplacian_var_threshold or lap_var > cfg.laplacian_var_max_threshold:
        spoof_votes.append("low_gradient_richness")
        logger.debug(f"Laplacian flag: {lap_var:.2f} < {cfg.laplacian_var_threshold}")

    # ── Decision ────────────────────────────────────────────────
    scores["spoof_votes"] = spoof_votes
    scores["total_votes"] = len(spoof_votes)

    logger.info(f"Spoof detection: votes={spoof_votes} scores={scores}")

    if len(spoof_votes) >= cfg.min_spoof_votes:
        reason_map = {
            "deepface_liveness":      "Liveness check failed",
            "fft_screen_pattern":     "Screen/display pattern detected (Moiré)",
            "lbp_texture_uniform":    "Flat texture detected (possible print attack)",
            "hsv_backlight":          "Abnormal brightness (possible screen/backlight)",
            "low_gradient_richness":  "Low image sharpness (re-photographed image)",
        }
        primary = spoof_votes[0]
        return SpoofResult(
            is_real=False,
            reason=reason_map.get(primary, "Spoof detected"),
            scores=scores
        )

    return SpoofResult(is_real=True, scores=scores)