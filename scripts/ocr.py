"""OCR for scanned leaflet pages and phone photos (plan 1.7). Needs Tesseract and tessdata\\aze.traineddata."""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from config import TESSDATA_DIR

_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR.as_posix()
if not shutil.which("tesseract") and _DEFAULT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_DEFAULT)


def _skew_angle(inv, max_angle=8.0, step=0.5):
    """Rotation (degrees) that makes the text lines sharpest."""
    small = cv2.resize(inv, (600, max(1, int(inv.shape[0] * 600 / inv.shape[1]))), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) > 3200:                                    # big phone photos: shrink for speed
        f = 3200 / max(h, w)
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    elif max(h, w) < 1800:                                  # small images: enlarge
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    angle = _skew_angle(inv)
    if abs(angle) > 0.3:
        hh, ww = gray.shape
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (ww, hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def ocr_array(img_bgr, psm=4):
    cfg = cfg = f"--oem 1 --psm {psm}"
    try:
        return pytesseract.image_to_string(preprocess(img_bgr), lang="aze", config=cfg)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError("Tesseract quraşdırılmayıb: winget install --id UB-Mannheim.TesseractOCR -e")
    except pytesseract.TesseractError as e:
        raise RuntimeError(f"Tesseract xətası (tessdata\\aze.traineddata var?): {e}")


def ocr_bytes(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Şəkil formatı tanınmadı")
    return ocr_array(img)