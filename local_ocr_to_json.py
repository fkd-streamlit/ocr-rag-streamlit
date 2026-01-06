# local_ocr_to_json.py
import json
from datetime import datetime
from pathlib import Path

import pytesseract
from PIL import Image
import pdf2image

# OpenCVがあれば前処理に使用（無ければPILで最低限）
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    np = None
    CV2_AVAILABLE = False

# ---- 設定 ----
PDF_DPI = 300
LANG = "jpn"       # "jpn+eng" も可
PSM = 6
OEM = 3

DEFAULT_CONTRAST = 1.0
DEFAULT_BRIGHTNESS = 0
DEFAULT_THRESHOLD = 127
DEFAULT_USE_ADAPTIVE = False

OUTPUT_DIR = Path("data/ocr_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_pil(image: Image.Image, contrast=1.0, brightness=0, threshold=127) -> Image.Image:
    from PIL import ImageEnhance
    img = image.convert("L")
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if brightness != 0:
        factor = max(0.1, 1.0 + brightness / 200.0)
        img = ImageEnhance.Brightness(img).enhance(factor)
    thr = max(0, min(255, int(threshold)))
    img = img.point(lambda p: 255 if p > thr else 0)
    return img.convert("RGB")


def preprocess(image: Image.Image, contrast=1.0, brightness=0, threshold=127, use_adaptive=False) -> Image.Image:
    if CV2_AVAILABLE and cv2 is not None and np is not None:
        cv_img = np.array(image.convert("RGB"))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        if contrast != 1.0 or brightness != 0:
            gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

        if use_adaptive:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            _, th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        return Image.fromarray(th).convert("RGB")

    return preprocess_pil(image, contrast=contrast, brightness=brightness, threshold=threshold)


def ocr_image(image: Image.Image) -> str:
    config = f"--oem {OEM} --psm {PSM} -l {LANG}"
    return pytesseract.image_to_string(image, config=config)


def now_id(prefix: str = "doc") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def process_file(path: Path,
                 contrast=DEFAULT_CONTRAST,
                 brightness=DEFAULT_BRIGHTNESS,
                 threshold=DEFAULT_THRESHOLD,
                 use_adaptive=DEFAULT_USE_ADAPTIVE) -> dict:
    ext = path.suffix.lower()
    texts = []

    if ext == ".pdf":
        images = pdf2image.convert_from_path(str(path), dpi=PDF_DPI)
        for page_idx, img in enumerate(images, start=1):
            proc = preprocess(img, contrast, brightness, threshold, use_adaptive)
            txt = ocr_image(proc).strip()
            if txt:
                texts.append(f"--- page {page_idx} ---\n{txt}")
    else:
        img = Image.open(path)
        proc = preprocess(img, contrast, brightness, threshold, use_adaptive)
        txt = ocr_image(proc).strip()
        if txt:
            texts.append(txt)

    full_text = "\n\n".join(texts)

    doc_id = now_id("doc")
    doc = {
        "id": doc_id,
        "filename": path.name,
        "text": full_text,
        "uploaded_at": datetime.now().isoformat(),
        "ocr_settings": {
            "source": "local_ocr",
            "lang": LANG,
            "psm": PSM,
            "oem": OEM,
            "pdf_dpi": PDF_DPI,
            "contrast": contrast,
            "brightness": brightness,
            "threshold": threshold,
            "use_adaptive": use_adaptive,
        },
    }
    return doc


def main():
    print("ローカルOCR → JSON化")
    target = input("PDF/画像ファイルのパスを入力: ").strip().strip('"')
    p = Path(target)
    if not p.exists():
        print("ファイルが見つかりません:", p)
        return

    doc = process_file(p)
    out_path = OUTPUT_DIR / f"{doc['id']}.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("保存しました:", out_path)
    print("抽出文字数:", len(doc["text"]))


if __name__ == "__main__":
    main()
