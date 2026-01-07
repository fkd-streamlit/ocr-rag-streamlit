
# -*- coding: utf-8 -*-
"""
ローカルOCR → JSONツール（安定版）
- 画像 / PDF を入力して OCR
- 出力: ./data/ocr_results/<元ファイル名>__YYYYmmdd_HHMMSS.json
使い方（例）:
  python local_ocr_to_json.py --pdf "C:/path/to/doc.pdf" --out "data/ocr_results/out.json" --dpi 200 --lang jpn+eng --start 1 --end 0
  python local_ocr_to_json.py --img "C:/path/to/image.png" --out "data/ocr_results/out.json" --lang jpn
"""
import os, sys, json, time, argparse, shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract

# pdf2image
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF2IMAGE_OK = True
except Exception:
    PDF2IMAGE_OK = False

# 設定ファイルから上限値を読み込み
try:
    import config
    MAX_PDF_FILE_SIZE_MB = getattr(config, 'MAX_PDF_FILE_SIZE_MB', 500)
    MAX_PDF_PAGES = getattr(config, 'MAX_PDF_PAGES', 1000)
except ImportError:
    MAX_PDF_FILE_SIZE_MB = 500
    MAX_PDF_PAGES = 1000

OUT_DIR_DEFAULT = Path("data") / "ocr_results"

# ---- Tesseract 検出 ----
def ensure_tesseract_cmd() -> Optional[str]:
    # 1) env
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return env_cmd
    # 2) Windows デフォルト
    win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in win_paths:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    # 3) PATH
    which = shutil.which("tesseract")
    if which and Path(which).exists():
        pytesseract.pytesseract.tesseract_cmd = which
        return which
    return None

# ---- 前処理（軽め）----
def preprocess_pil(img: Image.Image, contrast: float = 1.3, sharpen: bool = True) -> Image.Image:
    x = img.convert("RGB")
    x = ImageOps.grayscale(x)
    x = ImageOps.autocontrast(x)
    if contrast and contrast != 1.0:
        x = ImageEnhance.Contrast(x).enhance(contrast)
    if sharpen:
        x = x.filter(ImageFilter.SHARPEN)
    return x

# ---- OCR ----
def ocr_image(img: Image.Image, lang: str = "jpn+eng", psm: int = 6, oem: int = 3) -> str:
    config = f"--oem {oem} --psm {psm} -l {lang}"
    return pytesseract.image_to_string(img, config=config)

# ---- PDF容量チェック ----
def check_pdf_limits(pdf_path: Path) -> Tuple[bool, str]:
    """
    PDFファイルの容量とページ数をチェック
    Returns: (is_valid, error_message)
    """
    # ファイルサイズチェック
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_PDF_FILE_SIZE_MB:
        return False, f"PDFファイルサイズが上限を超えています: {file_size_mb:.1f}MB > {MAX_PDF_FILE_SIZE_MB}MB"
    
    # ページ数チェック
    if not PDF2IMAGE_OK:
        return False, "pdf2image がありません。pip install pdf2image。Windowsは poppler の PATH 設定も必要です。"
    
    try:
        info = pdfinfo_from_path(str(pdf_path))
        total_pages = int(info.get("Pages", 0))
        if total_pages > MAX_PDF_PAGES:
            return False, f"PDFページ数が上限を超えています: {total_pages}ページ > {MAX_PDF_PAGES}ページ"
        return True, f"OK: {file_size_mb:.1f}MB, {total_pages}ページ"
    except Exception as e:
        return False, f"PDF情報の取得に失敗しました: {e}"

# ---- PDF → 逐次画像化（メモリ節約）----
def iter_pdf_pages(pdf_path: Path, dpi: int, start_page: int, end_page: int):
    if not PDF2IMAGE_OK:
        raise RuntimeError("pdf2image がありません。pip install pdf2image。Windowsは poppler の PATH 設定も必要です。")
    
    # 容量チェック
    is_valid, msg = check_pdf_limits(pdf_path)
    if not is_valid:
        raise ValueError(msg)
    print(f"📄 {msg}")
    
    info = pdfinfo_from_path(str(pdf_path))
    total = int(info.get("Pages", 0))
    last = end_page if end_page and end_page > 0 else total
    first = max(1, start_page)
    
    print(f"🔄 処理範囲: {first}ページ目 ～ {last}ページ目（全{total}ページ）")
    
    for p in range(first, last + 1):
        # 1ページずつ変換（逐次）
        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=p, last_page=p)
        for img in images:
            yield p, img
        # 進捗表示（10ページごと）
        if (p - first + 1) % 10 == 0 or p == last:
            print(f"  ✓ {p - first + 1}/{last - first + 1}ページ処理完了")

# ---- メイン処理 ----
def process_image(img_path: Path, lang: str, psm: int, oem: int) -> Dict[str, Any]:
    img = Image.open(img_path)
    proc = preprocess_pil(img)
    text = ocr_image(proc, lang=lang, psm=psm, oem=oem)
    return {
        "doc_id": img_path.stem,
        "title": img_path.name,
        "source": str(img_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pages": [{
            "page_num": 1,
            "text": text,
            "metadata": {"dpi": None, "lang": lang, "preprocess": ["grayscale", "autocontrast", "sharpen"]}
        }]
    }

def process_pdf(pdf_path: Path, out_json: Path, dpi: int, lang: str, psm: int, oem: int, start_page: int, end_page: int) -> Dict[str, Any]:
    started = time.time()
    pages = []
    processed_count = 0
    
    try:
        for p_no, img in iter_pdf_pages(pdf_path, dpi=dpi, start_page=start_page, end_page=end_page):
            proc = preprocess_pil(img)
            text = ocr_image(proc, lang=lang, psm=psm, oem=oem)
            pages.append({
                "page_num": p_no,
                "text": text,
                "metadata": {"dpi": dpi, "lang": lang, "preprocess": ["grayscale", "autocontrast", "sharpen"]}
            })
            processed_count += 1
            
            # 逐次保存（長時間のクラッシュ対策／任意）
            if len(pages) % 50 == 0:
                _save_json_partial(out_json, pdf_path, pages)
                print(f"  💾 中間保存: {len(pages)}ページ")
    except MemoryError:
        # メモリ不足の場合、これまで処理した分を保存
        if pages:
            _save_json_partial(out_json, pdf_path, pages)
        raise RuntimeError(f"メモリ不足: {processed_count}ページまで処理しましたが、それ以上処理できませんでした。")
    except Exception as e:
        # エラー時もこれまで処理した分を保存
        if pages:
            _save_json_partial(out_json, pdf_path, pages)
        raise

    return {
        "doc_id": pdf_path.stem,
        "title": pdf_path.name,
        "source": str(pdf_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pages": pages,
        "elapsed_sec": round(time.time() - started, 3)
    }

def _save_json_partial(out_path: Path, src_path: Path, pages: List[Dict[str, Any]]):
    # 任意のスナップショット保存（中断リスク低減）
    payload = {
        "doc_id": src_path.stem,
        "title": src_path.name,
        "source": str(src_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pages": pages
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ---- CLI ----
def main():
    tcmd = ensure_tesseract_cmd()
    if not tcmd:
        print("❌ Tesseract を検出できません。環境変数 TESSERACT_CMD を設定するか、PATH を確認してください。")
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="OCR対象PDF")
    ap.add_argument("--img", help="OCR対象画像（PNG/JPG/TIFF など）")
    ap.add_argument("--out", default=str(OUT_DIR_DEFAULT / "result.json"), help="出力JSON")
    ap.add_argument("--dpi", type=int, default=200, help="PDF→画像変換DPI（150–200推奨）")
    ap.add_argument("--lang", default="jpn+eng", help="Tesseract言語（jpn / eng / jpn+eng）")
    ap.add_argument("--psm", type=int, default=6, help="Page Segmentation Mode")
    ap.add_argument("--oem", type=int, default=3, help="OCR Engine Mode")
    ap.add_argument("--start", type=int, default=1, help="開始ページ")
    ap.add_argument("--end", type=int, default=0, help="終了ページ（0=最後まで）")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.img:
            img_path = Path(args.img)
            if not img_path.exists():
                print(f"❌ 画像ファイルが見つかりません: {img_path}")
                sys.exit(1)
            payload = process_image(img_path, lang=args.lang, psm=args.psm, oem=args.oem)
        elif args.pdf:
            pdf_path = Path(args.pdf)
            if not pdf_path.exists():
                print(f"❌ PDFファイルが見つかりません: {pdf_path}")
                sys.exit(1)
            
            # PDF容量チェック
            is_valid, msg = check_pdf_limits(pdf_path)
            if not is_valid:
                print(f"❌ {msg}")
                print(f"💡 ヒント: 上限値を変更する場合は config.py の MAX_PDF_FILE_SIZE_MB または MAX_PDF_PAGES を調整してください。")
                sys.exit(1)
            
            payload = process_pdf(pdf_path, out_path, dpi=args.dpi, lang=args.lang, psm=args.psm, oem=args.oem,
                                  start_page=args.start, end_page=args.end)
        else:
            print("❌ --pdf または --img を指定してください。")
            sys.exit(1)

        # JSONファイルサイズチェック
        json_size_mb = len(json.dumps(payload, ensure_ascii=False)) / (1024 * 1024)
        try:
            max_json_size = getattr(config, 'MAX_JSON_FILE_SIZE_MB', 100)
            if json_size_mb > max_json_size:
                print(f"⚠️  警告: JSONファイルサイズが大きいです: {json_size_mb:.1f}MB（上限: {max_json_size}MB）")
        except:
            pass
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {out_path} ({json_size_mb:.1f}MB, {len(payload.get('pages', []))}ページ)")
    except ValueError as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)
    except MemoryError as e:
        print(f"❌ メモリ不足エラー: {e}")
        print(f"💡 ヒント: PDFを分割して処理するか、DPIを下げてみてください（--dpi 150など）。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

