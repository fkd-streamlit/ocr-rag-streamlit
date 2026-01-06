# -*- coding: utf-8 -*-
"""
ローカルOCR → JSON化ツール（完全版）
- 画像 / PDF を入力してOCR
- 出力: ./data/ocr_results/<ファイル名>__YYYYmmdd_HHMMSS.json
- 使い方:
    python .\local_ocr_to_json.py "C:\path\to\image_or_pdf"
    python .\local_ocr_to_json.py "C:\path\to\folder"      (フォルダ一括)
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

# PDF → 画像化（pdf2image）
try:
    import pdf2image
    PDF2IMAGE_OK = True
except Exception:
    PDF2IMAGE_OK = False


# =========================
# 設定（必要なら変更）
# =========================
DEFAULT_LANG = "jpn+eng"     # 日本語+英語
DEFAULT_PSM = 6
DEFAULT_OEM = 3

# 前処理（軽め）
DEFAULT_CONTRAST = 1.4      # 1.0が基準、上げるとくっきり
DEFAULT_SHARPEN = True

# 出力先
OUT_DIR = Path("data") / "ocr_results"


# =========================
# 共通ユーティリティ
# =========================
def ensure_tesseract_cmd():
    """
    pytesseract が tesseract.exe を見つけられるように設定。
    優先順位:
      1) 環境変数 TESSERACT_CMD
      2) 既定パス C:\Program Files\Tesseract-OCR\tesseract.exe
      3) PATH（shutil.which）
    """
    # 1) env
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return env_cmd

    # 2) default
    default_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(default_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = default_cmd
        return default_cmd

    # 3) PATH
    import shutil
    which_cmd = shutil.which("tesseract")
    if which_cmd and Path(which_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = which_cmd
        return which_cmd

    return None


def preprocess_pil(img: Image.Image,
                   contrast: float = DEFAULT_CONTRAST,
                   sharpen: bool = DEFAULT_SHARPEN) -> Image.Image:
    """
    OpenCV無しで前処理（Streamlit Cloudでも同じロジックにしやすい）
    - グレースケール
    - オートコントラスト
    - コントラスト調整
    - （任意）軽いシャープ
    """
    x = img.convert("RGB")
    x = ImageOps.grayscale(x)
    x = ImageOps.autocontrast(x)

    if contrast and contrast != 1.0:
        x = ImageEnhance.Contrast(x).enhance(contrast)

    if sharpen:
        # ほんのりシャープ（過剰にしない）
        x = x.filter(ImageFilter.SHARPEN)

    return x


def ocr_image(img: Image.Image,
              lang: str = DEFAULT_LANG,
              psm: int = DEFAULT_PSM,
              oem: int = DEFAULT_OEM) -> str:
    config = f'--oem {oem} --psm {psm} -l {lang}'
    return pytesseract.image_to_string(img, config=config)


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    if not PDF2IMAGE_OK:
        raise RuntimeError("pdf2image が入っていません。requirements に pdf2image を入れてください。")
    # Windowsはpopplerが必要な場合あり（READMEに記載済み）
    return pdf2image.convert_from_path(str(pdf_path), dpi=dpi)


def save_json(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def is_pdf_file(p: Path) -> bool:
    return p.suffix.lower() == ".pdf"


# =========================
# メイン処理
# =========================
def process_one_file(input_path: Path) -> Dict[str, Any]:
    """
    1ファイル（画像 or PDF）をOCRしてJSON用dictにして返す
    """
    started = time.time()
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    result: Dict[str, Any] = {
        "source_path": str(input_path),
        "processed_at": now,
        "tesseract_cmd": getattr(pytesseract.pytesseract, "tesseract_cmd", None),
        "lang": DEFAULT_LANG,
        "psm": DEFAULT_PSM,
        "oem": DEFAULT_OEM,
        "pages": [],
    }

    if is_image_file(input_path):
        img = Image.open(input_path)
        proc = preprocess_pil(img)
        text = ocr_image(proc)

        result["pages"].append({
            "page": 1,
            "text": text,
            "char_count": len(text),
        })

    elif is_pdf_file(input_path):
        imgs = pdf_to_images(input_path, dpi=300)
        for i, img in enumerate(imgs, start=1):
            proc = preprocess_pil(img)
            text = ocr_image(proc)
            result["pages"].append({
                "page": i,
                "text": text,
                "char_count": len(text),
            })
    else:
        raise ValueError(f"未対応のファイル形式です: {input_path.name}")

    # まとめ
    all_text = "\n\n".join(p["text"] for p in result["pages"])
    result["text"] = all_text
    result["total_char_count"] = len(all_text)
    result["elapsed_sec"] = round(time.time() - started, 3)

    return result


def process_path(path: Path) -> List[Path]:
    """
    入力がフォルダなら中の画像/PDFを列挙、ファイルならその1件
    """
    if path.is_file():
        return [path]

    if path.is_dir():
        files = []
        for p in sorted(path.rglob("*")):
            if p.is_file() and (is_image_file(p) or is_pdf_file(p)):
                files.append(p)
        return files

    raise FileNotFoundError(f"パスが見つかりません: {path}")


def main():
    # tesseract設定
    tcmd = ensure_tesseract_cmd()
    if not tcmd:
        print("❌ tesseract.exe が見つかりません。TESSERACT_CMD か PATH を確認してください。")
        sys.exit(1)

    # 引数 or 入力
    if len(sys.argv) >= 2:
        raw = sys.argv[1]
    else:
        raw = input("PDF/画像ファイルのパスを入力: ")

    raw = raw.strip().strip('"')
    target = Path(raw)

    targets = process_path(target)
    if not targets:
        print("対象ファイルが見つかりませんでした（画像/PDFが0件）。")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"ローカルOCR → JSON化")
    print(f"tesseract: {tcmd}")
    print(f"対象: {target}")
    print(f"件数: {len(targets)}")
    print("----")

    ok = 0
    ng = 0

    for p in targets:
        try:
            doc = process_one_file(p)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"{p.stem}__{ts}.json"
            out_path = OUT_DIR / out_name
            save_json(out_path, doc)
            ok += 1
            print(f"✅ {p.name} -> {out_path}")
        except Exception as e:
            ng += 1
            print(f"❌ {p.name} : {e}")

    print("----")
    print(f"完了: 成功 {ok} / 失敗 {ng}")
    print(f"出力先: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    # PILのSHARPENを使うため（忘れがち）
    from PIL import ImageFilter
    main()

