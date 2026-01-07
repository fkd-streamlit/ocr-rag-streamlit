# app.py  (Route A: ローカルでOCR→JSON化→検索) / Streamlit
# - data/ocr_results/*.json を自動読み込みして検索
# - クエリに応じて結果が変わる（バグ修正版）
# - キーワード(BM25風) + TF-IDF の両対応（scikit-learn使用）
#
# 注意: このファイルはStreamlit Cloudで実行されます
# - pdf2imageやpytesseractは使用しません（OCRはローカルで実行）
# - 必要なライブラリ: streamlit, scikit-learn のみ

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st

# OCR関連のインポート（オプション）
try:
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    import pytesseract
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False


# ----------------------------
# 設定
# ----------------------------
OCR_RESULTS_DIR = Path("data") / "ocr_results"

DEFAULT_TOPK = 5

# 容量上限設定（デフォルト値）
# config.pyから読み込もうとしますが、存在しない場合はデフォルト値を使用
MAX_JSON_FILE_SIZE_MB = 100
MAX_TOTAL_CHUNKS = 50000
MAX_PDF_FILE_SIZE_MB = 500
MAX_PDF_PAGES = 1000

# config.pyが存在する場合は上書き（オプション）
try:
    import config
    MAX_JSON_FILE_SIZE_MB = getattr(config, 'MAX_JSON_FILE_SIZE_MB', MAX_JSON_FILE_SIZE_MB)
    MAX_TOTAL_CHUNKS = getattr(config, 'MAX_TOTAL_CHUNKS', MAX_TOTAL_CHUNKS)
    MAX_PDF_FILE_SIZE_MB = getattr(config, 'MAX_PDF_FILE_SIZE_MB', MAX_PDF_FILE_SIZE_MB)
    MAX_PDF_PAGES = getattr(config, 'MAX_PDF_PAGES', MAX_PDF_PAGES)
except (ImportError, AttributeError, Exception):
    # config.pyが存在しない、またはエラーが発生した場合はデフォルト値を使用
    pass


# ----------------------------
# OCR機能（オプション）
# ----------------------------
def check_tesseract_available() -> Tuple[bool, str]:
    """Tesseractが利用可能かチェック"""
    if not PIL_OK:
        return False, "PIL/Pillowがインストールされていません"
    try:
        pytesseract.get_tesseract_version()
        return True, "Tesseract利用可能"
    except Exception as e:
        return False, f"Tesseractが見つかりません: {e}"

def preprocess_pil(img: Image.Image, contrast: float = 1.3, sharpen: bool = True) -> Image.Image:
    """画像前処理"""
    if not PIL_OK:
        raise ImportError("PIL/Pillowがインストールされていません")
    x = img.convert("RGB")
    x = ImageOps.grayscale(x)
    x = ImageOps.autocontrast(x)
    if contrast and contrast != 1.0:
        x = ImageEnhance.Contrast(x).enhance(contrast)
    if sharpen:
        x = x.filter(ImageFilter.SHARPEN)
    return x

def ocr_image(img: Image.Image, lang: str = "jpn+eng", psm: int = 6, oem: int = 3) -> str:
    """画像からOCR実行"""
    if not PIL_OK:
        raise ImportError("PIL/Pillowがインストールされていません")
    config_str = f"--oem {oem} --psm {psm} -l {lang}"
    return pytesseract.image_to_string(img, config=config_str)

def check_pdf_limits(pdf_bytes: bytes) -> Tuple[bool, str]:
    """PDFファイルの容量をチェック"""
    file_size_mb = len(pdf_bytes) / (1024 * 1024)
    if file_size_mb > MAX_PDF_FILE_SIZE_MB:
        return False, f"PDFファイルサイズが上限を超えています: {file_size_mb:.1f}MB > {MAX_PDF_FILE_SIZE_MB}MB"
    
    if not PDF2IMAGE_OK:
        return False, "pdf2imageがインストールされていません"
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)
        
        try:
            info = pdfinfo_from_path(str(tmp_path))
            total_pages = int(info.get("Pages", 0))
            if total_pages > MAX_PDF_PAGES:
                return False, f"PDFページ数が上限を超えています: {total_pages}ページ > {MAX_PDF_PAGES}ページ"
            return True, f"OK: {file_size_mb:.1f}MB, {total_pages}ページ"
        finally:
            tmp_path.unlink()
    except Exception as e:
        return False, f"PDF情報の取得に失敗しました: {e}"

def process_pdf_upload(pdf_bytes: bytes, filename: str, dpi: int = 200, lang: str = "jpn+eng", 
                       psm: int = 6, oem: int = 3, progress_callback=None) -> Dict[str, Any]:
    """アップロードされたPDFをOCR処理"""
    if not PDF2IMAGE_OK:
        raise RuntimeError("pdf2imageがインストールされていません。pip install pdf2image")
    if not PIL_OK:
        raise RuntimeError("PIL/Pillowがインストールされていません")
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    
    try:
        started = time.time()
        pages = []
        
        # PDF情報取得
        info = pdfinfo_from_path(str(tmp_path))
        total_pages = int(info.get("Pages", 0))
        
        # 1ページずつ処理
        for p_no in range(1, total_pages + 1):
            images = convert_from_path(str(tmp_path), dpi=dpi, first_page=p_no, last_page=p_no)
            for img in images:
                proc = preprocess_pil(img)
                text = ocr_image(proc, lang=lang, psm=psm, oem=oem)
                pages.append({
                    "page": p_no,
                    "text": text,
                    "metadata": {"dpi": dpi, "lang": lang, "preprocess": ["grayscale", "autocontrast", "sharpen"]}
                })
                
                if progress_callback:
                    progress_callback(p_no, total_pages)
        
        return {
            "doc_id": Path(filename).stem,
            "title": filename,
            "source": filename,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "pages": pages,
            "elapsed_sec": round(time.time() - started, 3)
        }
    finally:
        tmp_path.unlink()

def process_image_upload(img_bytes: bytes, filename: str, lang: str = "jpn+eng", 
                        psm: int = 6, oem: int = 3) -> Dict[str, Any]:
    """アップロードされた画像をOCR処理"""
    if not PIL_OK:
        raise RuntimeError("PIL/Pillowがインストールされていません")
    
    import io
    img = Image.open(io.BytesIO(img_bytes))
    proc = preprocess_pil(img)
    text = ocr_image(proc, lang=lang, psm=psm, oem=oem)
    
    return {
        "doc_id": Path(filename).stem,
        "title": filename,
        "source": filename,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pages": [{
            "page": 1,
            "text": text,
            "metadata": {"dpi": None, "lang": lang, "preprocess": ["grayscale", "autocontrast", "sharpen"]}
        }]
    }


# ----------------------------
# ユーティリティ
# ----------------------------
def normalize_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_tokenize_ja(s: str) -> List[str]:
    """
    形態素解析なしの簡易トークナイズ。
    - 日本語/英数字をそれっぽく分割して、TF-IDFやキーワード検索の入力にする
    """
    s = normalize_text(s).lower()
    # ひらがな/カタカナ/漢字/英数字をまとめて拾う
    tokens = re.findall(r"[一-龥]+|[ぁ-ん]+|[ァ-ヴー]+|[a-z0-9]+", s)
    # 1文字だけはノイズになりやすいので除外（必要なら外してください）
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens


def load_ocr_json(path: Path) -> List[Dict[str, Any]]:
    """
    local_ocr_to_json.py が出力する想定のJSON:
    { "meta":..., "pages":[{"page":1,"text":"..."}, ...] }
    もし形式が違っても、pages/text をできるだけ拾う
    """
    # JSONファイルサイズチェック
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_JSON_FILE_SIZE_MB:
        raise ValueError(f"JSONファイルサイズが上限を超えています: {path.name} ({file_size_mb:.1f}MB > {MAX_JSON_FILE_SIZE_MB}MB)")
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONファイルの解析に失敗しました: {path.name} - {e}")
    except MemoryError:
        raise MemoryError(f"JSONファイルが大きすぎてメモリに読み込めません: {path.name} ({file_size_mb:.1f}MB)")

    pages = []
    if isinstance(data, dict) and "pages" in data and isinstance(data["pages"], list):
        for p in data["pages"]:
            txt = p.get("text", "")
            page_no = p.get("page", None)
            pages.append({"page": page_no, "text": normalize_text(txt)})
        return pages

    # フォールバック（想定外形式）
    if isinstance(data, list):
        for i, p in enumerate(data, start=1):
            if isinstance(p, dict) and "text" in p:
                pages.append({"page": p.get("page", i), "text": normalize_text(p.get("text", ""))})
    return pages


@dataclass
class Chunk:
    doc_id: str
    source_file: str
    page: int | None
    chunk_id: int
    text: str


def make_chunks(doc_id: str, source_file: str, pages: List[Dict[str, Any]], chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0
    for p in pages:
        page_no = p.get("page", None)
        text = p.get("text", "")
        if not text:
            continue

        # ページごとにスライドチャンク
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            ctext = text[start:end]
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    source_file=source_file,
                    page=page_no,
                    chunk_id=cid,
                    text=ctext,
                )
            )
            cid += 1
            if end == len(text):
                break
            start = max(0, end - overlap)
    return chunks


def load_all_chunks(ocr_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not ocr_dir.exists():
        return chunks

    json_files = sorted(ocr_dir.glob("*.json"))
    if not json_files:
        return chunks
    
    skipped_files = []
    for jp in json_files:
        try:
            pages = load_ocr_json(jp)
            doc_id = jp.stem
            new_chunks = make_chunks(doc_id=doc_id, source_file=jp.name, pages=pages)
            
            # 総チャンク数チェック
            if len(chunks) + len(new_chunks) > MAX_TOTAL_CHUNKS:
                skipped_files.append(f"{jp.name} (チャンク数上限に達しました: {len(chunks) + len(new_chunks)} > {MAX_TOTAL_CHUNKS})")
                break
            
            chunks.extend(new_chunks)
        except (ValueError, MemoryError) as e:
            skipped_files.append(f"{jp.name} ({str(e)})")
            continue
    
    if skipped_files:
        import warnings
        warnings.warn(f"以下のファイルをスキップしました:\n" + "\n".join(f"  - {f}" for f in skipped_files))
    
    return chunks


# ----------------------------
# 検索（1）キーワードスコア（簡易BM25風）
# ----------------------------
def keyword_score(query: str, text: str) -> float:
    q_tokens = simple_tokenize_ja(query)
    if not q_tokens:
        return 0.0
    t = normalize_text(text).lower()
    score = 0.0
    for tok in q_tokens:
        # 出現回数を加点（軽い重み）
        c = t.count(tok)
        if c > 0:
            score += 1.0 + min(3.0, c * 0.3)
    return score


# ----------------------------
# 検索（2）TF-IDF
# ----------------------------
@st.cache_resource(show_spinner=False)
def build_tfidf_index(texts: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenize_ja,
        lowercase=True,
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)

    def search(q: str) -> List[float]:
        qv = vectorizer.transform([q])
        sims = cosine_similarity(qv, X).flatten()
        return sims.tolist()

    return search


def search_chunks(chunks: List[Chunk], query: str, topk: int = 5) -> List[Tuple[float, Chunk]]:
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    tfidf_search = build_tfidf_index(texts)
    tfidf_scores = tfidf_search(query)

    # キーワードも混ぜて最終スコア
    scored: List[Tuple[float, Chunk]] = []
    for s, c in zip(tfidf_scores, chunks):
        ks = keyword_score(query, c.text)
        final = float(s) * 1.0 + ks * 0.25  # ←混合比率（必要なら調整）
        scored.append((final, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    # スコアがほぼゼロのものは除外（ただし全部ゼロならトップを返す）
    if scored and scored[0][0] <= 1e-8:
        return scored[:topk]
    return [x for x in scored[:topk] if x[0] > 1e-8] or scored[:topk]


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="OCR RAG (Local)", page_icon="📄", layout="wide")
st.title("📄 OCR RAG（ローカルOCR→JSON→検索）")

with st.sidebar:
    st.header("設定")
    st.write("検索対象： `data/ocr_results/*.json`")
    
    # PDF/画像アップロードとOCR実行機能
    st.subheader("📄 PDF/画像をアップロードしてOCR")
    ocr_available, ocr_msg = check_tesseract_available()
    
    if not ocr_available:
        st.warning(f"⚠️ OCR機能は利用できません: {ocr_msg}")
        st.info("💡 ローカルで実行する場合は、以下をインストールしてください:\n- Tesseract OCR\n- Poppler (PDF用)\n- pip install pillow pdf2image pytesseract")
    else:
        st.success(f"✅ {ocr_msg}")
        
        uploaded_pdf = st.file_uploader(
            "PDFファイルをアップロード",
            type=["pdf"],
            help="PDFファイルをアップロードしてOCRを実行します"
        )
        
        uploaded_image = st.file_uploader(
            "画像ファイルをアップロード",
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            help="画像ファイルをアップロードしてOCRを実行します"
        )
        
        if uploaded_pdf or uploaded_image:
            # OCR設定
            with st.expander("OCR設定", expanded=False):
                dpi = st.number_input("DPI (PDF用)", min_value=100, max_value=600, value=200, step=50)
                lang = st.selectbox("言語", ["jpn", "jpn+eng", "eng"], index=1)
                psm = st.number_input("PSM (Page Segmentation Mode)", min_value=0, max_value=13, value=6)
                oem = st.number_input("OEM (OCR Engine Mode)", min_value=0, max_value=3, value=3)
            
            if st.button("🚀 OCR実行", type="primary"):
                if uploaded_pdf:
                    with st.spinner("PDFを処理中..."):
                        try:
                            pdf_bytes = uploaded_pdf.getvalue()
                            # 容量チェック
                            is_valid, msg = check_pdf_limits(pdf_bytes)
                            if not is_valid:
                                st.error(f"❌ {msg}")
                            else:
                                st.info(f"📄 {msg}")
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                def progress_callback(current, total):
                                    progress_bar.progress(current / total)
                                    status_text.text(f"処理中: {current}/{total}ページ")
                                
                                result = process_pdf_upload(
                                    pdf_bytes, uploaded_pdf.name, dpi=dpi, 
                                    lang=lang, psm=psm, oem=oem,
                                    progress_callback=progress_callback
                                )
                                
                                # JSONとして保存
                                json_filename = f"{Path(uploaded_pdf.name).stem}.json"
                                save_path = OCR_RESULTS_DIR / json_filename
                                OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                                
                                with save_path.open("w", encoding="utf-8") as f:
                                    json.dump(result, f, ensure_ascii=False, indent=2)
                                
                                progress_bar.empty()
                                status_text.empty()
                                st.success(f"✅ OCR完了: {len(result['pages'])}ページ → {json_filename}")
                                
                                # セッション状態をリセット
                                if "chunks" in st.session_state:
                                    del st.session_state["chunks"]
                        except Exception as e:
                            st.error(f"❌ OCR処理エラー: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                
                if uploaded_image:
                    with st.spinner("画像を処理中..."):
                        try:
                            img_bytes = uploaded_image.getvalue()
                            result = process_image_upload(
                                img_bytes, uploaded_image.name,
                                lang=lang, psm=psm, oem=oem
                            )
                            
                            # JSONとして保存
                            json_filename = f"{Path(uploaded_image.name).stem}.json"
                            save_path = OCR_RESULTS_DIR / json_filename
                            OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                            
                            with save_path.open("w", encoding="utf-8") as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                            
                            st.success(f"✅ OCR完了: {json_filename}")
                            
                            # セッション状態をリセット
                            if "chunks" in st.session_state:
                                del st.session_state["chunks"]
                        except Exception as e:
                            st.error(f"❌ OCR処理エラー: {e}")
                            import traceback
                            st.code(traceback.format_exc())
    
    st.divider()
    
    # JSONファイルアップロード機能
    st.subheader("📤 JSONファイルをアップロード")
    uploaded_files = st.file_uploader(
        "OCR結果のJSONファイルをアップロード",
        type=["json"],
        accept_multiple_files=True,
        help="local_ocr_to_json.pyで生成したJSONファイルをアップロードしてください"
    )
    
    if uploaded_files:
        OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        for uploaded_file in uploaded_files:
            try:
                # ファイルサイズチェック
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size_mb > MAX_JSON_FILE_SIZE_MB:
                    st.warning(f"⚠️ {uploaded_file.name} はサイズが大きすぎます ({file_size_mb:.1f}MB > {MAX_JSON_FILE_SIZE_MB}MB)")
                    continue
                
                # ファイルを保存
                save_path = OCR_RESULTS_DIR / uploaded_file.name
                with save_path.open("wb") as f:
                    f.write(uploaded_file.getvalue())
                saved_count += 1
                st.success(f"✅ {uploaded_file.name} を保存しました ({file_size_mb:.1f}MB)")
            except Exception as e:
                st.error(f"❌ {uploaded_file.name} の保存に失敗しました: {e}")
        
        if saved_count > 0:
            st.info(f"ℹ️ {saved_count}個のファイルを保存しました。「JSONを再読み込み」ボタンをクリックしてください。")
            # セッション状態をリセットして再読み込みを促す
            if "chunks" in st.session_state:
                del st.session_state["chunks"]
    
    st.divider()
    
    topk = st.slider("表示件数 (Top-K)", 1, 10, DEFAULT_TOPK)
    chunk_size = st.number_input("チャンクサイズ", min_value=300, max_value=3000, value=900, step=100)
    overlap = st.number_input("オーバーラップ", min_value=0, max_value=500, value=150, step=10)
    reload_btn = st.button("🔄 JSONを再読み込み")

# 読み込み
if "chunks" not in st.session_state or reload_btn:
    with st.spinner("JSONファイルを読み込み中..."):
        try:
            raw_chunks = load_all_chunks(OCR_RESULTS_DIR)

            # チャンク設定変更に対応（再分割したい）
            # いったん pages を読み直して作り直す
            rebuilt: List[Chunk] = []
            skipped_count = 0
            
            json_files = sorted(OCR_RESULTS_DIR.glob("*.json"))
            progress_bar = st.progress(0)
            for idx, jp in enumerate(json_files):
                try:
                    pages = load_ocr_json(jp)
                    new_chunks = make_chunks(jp.stem, jp.name, pages, chunk_size=int(chunk_size), overlap=int(overlap))
                    
                    # 総チャンク数チェック
                    if len(rebuilt) + len(new_chunks) > MAX_TOTAL_CHUNKS:
                        st.warning(f"⚠️ チャンク数上限に達したため、{jp.name} 以降のファイルをスキップしました。")
                        skipped_count = len(json_files) - idx
                        break
                    
                    rebuilt.extend(new_chunks)
                    progress_bar.progress((idx + 1) / len(json_files))
                except (ValueError, MemoryError) as e:
                    st.warning(f"⚠️ {jp.name} をスキップしました: {e}")
                    skipped_count += 1
                    continue
            
            progress_bar.empty()
            st.session_state["chunks"] = rebuilt
            
            if skipped_count > 0:
                st.info(f"ℹ️ {skipped_count}個のファイルをスキップしました。容量上限を調整する場合は config.py を編集してください。")
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {e}")
            st.session_state["chunks"] = []

chunks: List[Chunk] = st.session_state["chunks"]

st.caption(f"読み込みJSON数: {len(list(OCR_RESULTS_DIR.glob('*.json')))} / チャンク数: {len(chunks)}")

if not chunks:
    st.warning("`data/ocr_results` に JSON がありません。")
    st.info("""
    **JSONファイルを追加する方法：**
    
    1. **アップロード機能を使用（推奨）**
       - 左サイドバーの「📤 JSONファイルをアップロード」からJSONファイルをアップロード
    
    2. **ローカルでOCRを実行**
       - `local_ocr_to_json.py` を使ってPDF/画像をOCRし、JSONを生成
       - 生成したJSONファイルを `data/ocr_results/` に配置
    
    3. **GitHubに配置（Streamlit Cloudの場合）**
       - JSONファイルを `data/ocr_results/` に配置してGitHubにプッシュ
    """)
    st.stop()

query = st.text_input("検索ワード（例：材料 / 定員 / 5052 / アルミニウム）", value="定員")
go = st.button("🔎 検索")

if go:
    results = search_chunks(chunks, query=query, topk=topk)

    st.subheader("検索結果")
    if not results:
        st.info("該当が見つかりませんでした（スコアが全てゼロ）。別の語や表記ゆれで試してください。")
    else:
        for score, c in results:
            header = f"Score: {score:.4f} | File: {c.source_file} | Doc: {c.doc_id}"
            if c.page is not None:
                header += f" | Page: {c.page}"
            with st.expander(header, expanded=False):
                # クエリをハイライト
                t = c.text
                if query.strip():
                    t = re.sub(re.escape(query.strip()), lambda m: f"**{m.group(0)}**", t)
                st.write(t)
