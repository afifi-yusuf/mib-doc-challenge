"""Fee-status recovery from noisy full-page scans via HOG + RandomForest.

Many gold-``paid`` APPROVED packets store the fee receipt only as a heavily
noised raster. Full-page Tesseract is slow and often fails; this module
classifies sliding-window crops in the header band.
"""

from __future__ import annotations

import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from skimage.feature import hog
except ImportError:  # pragma: no cover
    hog = None  # type: ignore

_MODEL_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "models" / "fee_hog_rf.pkl",
    Path("/app/models/fee_hog_rf.pkl"),
    Path("models/fee_hog_rf.pkl"),
]


@lru_cache(maxsize=1)
def _load_model() -> dict[str, Any] | None:
    if hog is None:
        return None
    for path in _MODEL_CANDIDATES:
        if path.is_file():
            with path.open("rb") as f:
                return pickle.load(f)
    return None


def _preprocess(gray: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def classify_fee_status(image_bgr: np.ndarray, min_margin: float = 1.5) -> tuple[str | None, float]:
    """Return (fee_status, confidence) or (None, 0) if uncertain."""
    model = _load_model()
    if model is None or image_bgr is None or image_bgr.size == 0:
        return None, 0.0
    clf = model["clf"]

    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    # Downscale large scans — fee header text remains readable at ~700px wide.
    h0, w0 = gray.shape[:2]
    if w0 > 800:
        scale = 800 / float(w0)
        gray = cv2.resize(gray, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    gray = _preprocess(gray)
    h, w = gray.shape
    # Header band only (fee status lives top-left on fee receipts).
    region = gray[: int(h * 0.32), : int(w * 0.65)]

    wins: list[np.ndarray] = []
    # Coarse stride for speed; two window sizes cover typical glyph aspect.
    for wh, ww in ((28, 64), (36, 88)):
        step_y = max(10, wh // 2)
        step_x = max(12, ww // 2)
        for y0 in range(0, max(1, region.shape[0] - wh), step_y):
            for x0 in range(0, max(1, region.shape[1] - ww), step_x):
                crop = cv2.resize(region[y0 : y0 + wh, x0 : x0 + ww], (64, 32))
                feat = hog(
                    crop,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    feature_vector=True,
                )
                wins.append(feat)
    if not wins:
        return None, 0.0

    probs = clf.predict_proba(np.asarray(wins))
    classes = list(clf.classes_)
    votes: Counter[str] = Counter()
    for p in probs:
        j = int(p.argmax())
        conf = float(p[j])
        if conf >= 0.55:
            votes[str(classes[j])] += conf
    if not votes:
        return None, 0.0
    ranked = votes.most_common()
    best_lab, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 3.0:
        return None, 0.0
    if second > 0 and best_score / second < min_margin:
        return None, 0.0
    conf = min(0.85, 0.55 + 0.025 * best_score)
    return best_lab, conf


def recover_fee_from_pixmap(pix) -> tuple[str | None, float]:
    """Accept a PyMuPDF Pixmap-like object."""
    import fitz

    if pix.n > 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return classify_fee_status(img)
