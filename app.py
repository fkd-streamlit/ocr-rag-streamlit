# app.py
# -----------------------------------------------
# OCR → テキスト化 → 検索（TF-IDF）までを “Streamlit Cloud 単体” で完結させる版
# - ユーザー側に Tesseract / Poppler のインストール不要（Cloud 側で packages.txt で入れる想定）
# - PDF / 画像アップロード → OCR → チャンク化 → 検索
# - サイドバーで OCR 精度（DPI/PSM/OEM/前処理）を調整可能
# -----------------------------------------------

from __future__ import annotations

import io
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# 画像処理
import cv2
from PIL import Image

# OCR / PDF
import pytesseract
from pdf2image import convert_from_bytes

# 検索（TF-IDF）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 基本設定
# =========================
APP_TITLE = "OCR RAG Search（PDF/画像 → OCR → 検索）"
DEFAULT_LANG = "jpn+eng"  # jpnのみでも可
DEFAULT_DPI = 250
DEFAULT_PSM = 6
DEFAULT_OEM = 3

# JSON（OCR結果）の保存先（Cloudでも動くよう相対パス）
DEFAULT_JSON_DIR = os.path.join("data", "ocr_results")


# =========================
# ユーティリティ
# =========================
def ensure_dirs() -> None:
    os.makedirs(DEFAULT_JSON_DIR, exist_ok=True)


def set_tesseract_cmd_if_needed() -> None:
    """
    Windowsローカル用：TESSERACT_CMD が指定されていればそれを使う。
    Streamlit Cloud では基本 PATH に tesseract が入る想定。
    """
    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def is_tesseract_available() -> Tuple[bool, str]:
    """
    tesseract が使えるか軽くチェック
    """
    try:
        v = pytesseract.get_tesseract_version()
        return True, f"Tesseract: {v}"
    except Exception as e:
        return False, f"Tesseractが利用できません: {e}"


def safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^\w\-\.\(\)ぁ-んァ-ン一-龥]+", "_", name)
    return name[:120] if len(name) > 120 else name


def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_for_ocr(
    img_bgr: np.ndarray,
    *,
    scale: float = 1.5,
    denoise: int = 1,
    contrast: int = 20,   # -100..100
    brightness: int = 0,  # -100..100
    binarize: bool = True,
    adaptive: bool = True,
    invert: bool = False,
    sharpen: bool = True,
) -> np.ndarray:
    """
    手書き/濃淡/スキャンのブレに対応しやすい前処理セット
    """
    h, w = img_bgr.shape[:2]
    if scale and scale != 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 明るさ・コントラスト調整
    # new = gray * (1 + contrast/100) + brightness
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    if denoise >= 1:
        # 軽いノイズ除去
        gray = cv2.medianBlur(gray, 3)
    if denoise >= 2:
        # もう少し強め
        gray = cv2.bilateralFilter(gray, 7, 50, 50)

    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)

    if binarize:
        if adaptive:
            th = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 10
            )
        else:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray = th

    if invert:
        gray = cv2.bitwise_not(gray)

    # pytesseract は PIL も受けられるが、ここでは BGR に戻す（表示にも使える）
    out_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out_bgr


def run_tesseract(
    img_bgr: np.ndarray,
    *,
    lang: str,
    psm: int,
    oem: int,
) -> str:
    pil = cv_to_pil(img_bgr)
    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(pil, lang=lang, config=config)
    return text.strip()


def pdf_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    """
    Poppler が入っていれば convert_from_bytes が動く（Streamlit Cloud では packages.txt で導入想定）
    """
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    return images


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks


def highlight_snippet(s: str, q: str, max_len: int = 350) -> str:
    s2 = " ".join(s.split())
    q = q.strip()
    if not q:
        return (s2[:max_len] + "…") if len(s2) > max_len else s2

    # クエリはスペース区切りの語も拾う
    terms = [t for t in re.split(r"\s+", q) if t]
    # まず最初の一致位置を探す
    pos = None
    for t in terms:
        m = re.search(re.escape(t), s2, flags=re.IGNORECASE)
        if m:
            pos = m.start()
            break

    if pos is None:
        return (s2[:max_len] + "…") if len(s2) > max_len else s2

    start = max(0, pos - max_len // 3)
    end = min(len(s2), start + max_len)
    snippet = s2[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(s2):
        snippet = snippet + "…"

    # 強調（太字）※markdown
    for t in sorted(terms, key=len, reverse=True):
        snippet = re.sub(
            re.escape(t),
            lambda m: f"**{m.group(0)}**",
            snippet,
            flags=re.IGNORECASE,
        )
    return snippet


# =========================
# TF-IDF インデックス
# =========================
@dataclass
class IndexedChunk:
    doc_name: str
    page: int
    chunk_id: int
    text: str


class TfIdfSearchIndex:
    def __init__(self) -> None:
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.items: List[IndexedChunk] = []

    def build(self, items: List[IndexedChunk]) -> None:
        self.items = items
        corpus = [it.text for it in items]
        self.vectorizer = TfidfVectorizer(
            lowercase=False,          # 日本語のため
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            max_features=80000,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 8) -> List[Tuple[IndexedChunk, float]]:
        if not self.items or self.vectorizer is None or self.matrix is None:
            return []
        q = query.strip()
        if not q:
            return []
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix).flatten()
        if sims.size == 0:
            return []
        idxs = np.argsort(-sims)[:top_k]
        results = [(self.items[i], float(sims[i])) for i in idxs]
        return results


# =========================
# ドキュメント形式（JSON）
# =========================
def make_doc_json(
    doc_name: str,
    pages_text: List[str],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": "ocr_doc_v1",
        "doc_name": doc_name,
        "meta": meta,
        "pages": [{"page": i + 1, "text": pages_text[i]} for i in range(len(pages_text))],
    }


def load_doc_json(file_bytes: bytes) -> Dict[str, Any]:
    return json.loads(file_bytes.decode("utf-8"))


def doc_to_index_items(doc: Dict[str, Any], chunk_size: int, overlap: int) -> List[IndexedChunk]:
    doc_name = doc.get("doc_name", "document")
    items: List[IndexedChunk] = []
    for p in doc.get("pages", []):
        page_no = int(p.get("page", 0))
        text = p.get("text", "") or ""
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for ci, ch in enumerate(chunks):
            items.append(IndexedChunk(doc_name=doc_name, page=page_no, chunk_id=ci, text=ch))
    return items


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
ensure_dirs()
set_tesseract_cmd_if_needed()

st.title(APP_TITLE)
st.caption("PDF/画像をアップロード → OCR → テキスト検索。Cloudで共有する前提の構成です。")

ok, tmsg = is_tesseract_available()
colA, colB = st.columns([1, 2])
with colA:
    st.write("**環境チェック**")
with colB:
    st.info(tmsg if ok else tmsg)

with st.sidebar:
    st.header("OCR 設定（精度調整）")

    lang = st.text_input("言語（Tesseract lang）", value=DEFAULT_LANG, help="例: jpn / jpn+eng")
    dpi = st.slider("PDF → 画像化 DPI", 150, 400, DEFAULT_DPI, 10)
    psm = st.selectbox("PSM（レイアウト）", options=[3, 4, 6, 11, 12], index=2, help="6=ブロック、11/12=疎なテキストに強め")
    oem = st.selectbox("OEM（エンジン）", options=[1, 3], index=1, help="3=既定（LSTM優先）")

    st.divider()
    st.subheader("前処理")
    scale = st.slider("拡大倍率", 1.0, 3.0, 1.6, 0.1)
    denoise = st.selectbox("ノイズ除去", options=[0, 1, 2], index=1, help="手書きやスキャンは 1〜2 が安定")
    contrast = st.slider("コントラスト", -50, 80, 25, 5)
    brightness = st.slider("明るさ", -50, 50, 0, 5)

    binarize = st.checkbox("二値化する", value=True)
    adaptive = st.checkbox("適応的二値化（濃淡に強い）", value=True)
    invert = st.checkbox("白黒反転（白文字/黒背景など）", value=False)
    sharpen = st.checkbox("シャープ化", value=True)

    st.divider()
    st.header("検索設定")
    chunk_size = st.slider("チャンクサイズ", 400, 1800, 900, 50)
    overlap = st.slider("オーバーラップ", 0, 400, 150, 10)
    top_k = st.slider("上位表示件数", 3, 20, 8, 1)
    min_score = st.slider("最小スコア（足切り）", 0.0, 1.0, 0.10, 0.01)

st.divider()

tab1, tab2 = st.tabs(["① アップロードしてOCR", "② OCR済みJSONを読み込み"])

# セッション
if "doc" not in st.session_state:
    st.session_state["doc"] = None
if "index" not in st.session_state:
    st.session_state["index"] = None
if "index_items" not in st.session_state:
    st.session_state["index_items"] = []

# ---- ① OCR ----
with tab1:
    st.subheader("PDF / 画像 をアップロードしてOCR")

    up = st.file_uploader(
        "PDFまたは画像（png/jpg）を選択",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    run_btn = st.button("OCR 実行", type="primary", disabled=(up is None))

    preview_col1, preview_col2 = st.columns([1, 1])

    if run_btn and up is not None:
        if not ok:
            st.error(
                "OCRエラー: tesseract が利用できません。\n"
                "Streamlit Cloud では packages.txt で tesseract-ocr と tesseract-ocr-jpn を入れてください。"
            )
        else:
            with st.spinner("OCR中...（ページ数やDPIにより時間がかかります）"):
                fname = safe_filename(up.name)
                b = up.read()

                pages_text: List[str] = []
                preview_images: List[Image.Image] = []

                if fname.lower().endswith(".pdf"):
                    try:
                        images = pdf_to_images(b, dpi=dpi)
                    except Exception as e:
                        st.error(f"PDFの画像化に失敗しました（Poppler未導入の可能性）: {e}")
                        st.stop()

                    for i, pil_img in enumerate(images, start=1):
                        img_bgr = pil_to_cv(pil_img)
                        pre = preprocess_for_ocr(
                            img_bgr,
                            scale=scale,
                            denoise=denoise,
                            contrast=contrast,
                            brightness=brightness,
                            binarize=binarize,
                            adaptive=adaptive,
                            invert=invert,
                            sharpen=sharpen,
                        )
                        text = run_tesseract(pre, lang=lang, psm=int(psm), oem=int(oem))
                        pages_text.append(text)
                        if i <= 2:
                            preview_images.append(cv_to_pil(pre))

                else:
                    pil_img = Image.open(io.BytesIO(b)).convert("RGB")
                    img_bgr = pil_to_cv(pil_img)
                    pre = preprocess_for_ocr(
                        img_bgr,
                        scale=scale,
                        denoise=denoise,
                        contrast=contrast,
                        brightness=brightness,
                        binarize=binarize,
                        adaptive=adaptive,
                        invert=invert,
                        sharpen=sharpen,
                    )
                    text = run_tesseract(pre, lang=lang, psm=int(psm), oem=int(oem))
                    pages_text = [text]
                    preview_images = [cv_to_pil(pre)]

                meta = {
                    "source": "upload",
                    "filename": fname,
                    "dpi": dpi,
                    "lang": lang,
                    "psm": int(psm),
                    "oem": int(oem),
                    "preprocess": {
                        "scale": scale,
                        "denoise": denoise,
                        "contrast": contrast,
                        "brightness": brightness,
                        "binarize": binarize,
                        "adaptive": adaptive,
                        "invert": invert,
                        "sharpen": sharpen,
                    },
                }

                doc = make_doc_json(doc_name=fname, pages_text=pages_text, meta=meta)
                st.session_state["doc"] = doc

                # インデックス作成
                items = doc_to_index_items(doc, chunk_size=chunk_size, overlap=overlap)
                idx = TfIdfSearchIndex()
                if items:
                    idx.build(items)
                st.session_state["index_items"] = items
                st.session_state["index"] = idx

            st.success("OCR完了＆検索インデックスを作成しました。")

            with preview_col1:
                st.write("**前処理プレビュー（最大2枚）**")
                for im in preview_images[:2]:
                    st.image(im, use_container_width=True)

            with preview_col2:
                st.write("**OCRテキスト（先頭ページ）**")
                st.text_area("",
                             value=(pages_text[0] if pages_text else ""),
                             height=260)

            # JSON保存 / ダウンロード
            st.write("### OCR結果（JSON）")
            json_bytes = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "JSONをダウンロード",
                data=json_bytes,
                file_name=f"{os.path.splitext(fname)[0]}.json",
                mime="application/json",
            )

# ---- ② JSON読み込み ----
with tab2:
    st.subheader("OCR済みJSON（ocr_doc_v1）を読み込み")
    jup = st.file_uploader("JSONを選択", type=["json"], accept_multiple_files=False, key="json_uploader")
    load_btn = st.button("JSON 読み込み", disabled=(jup is None))

    if load_btn and jup is not None:
        try:
            doc = load_doc_json(jup.read())
            if doc.get("schema") != "ocr_doc_v1":
                st.warning("schema が ocr_doc_v1 ではありません（読み込みは継続します）。")
            st.session_state["doc"] = doc

            items = doc_to_index_items(doc, chunk_size=chunk_size, overlap=overlap)
            idx = TfIdfSearchIndex()
            if items:
                idx.build(items)
            st.session_state["index_items"] = items
            st.session_state["index"] = idx

            st.success("JSONを読み込み、検索インデックスを作成しました。")
        except Exception as e:
            st.error(f"JSON読み込みに失敗: {e}")

st.divider()

# =========================
# 検索UI
# =========================
doc = st.session_state.get("doc")
idx: TfIdfSearchIndex = st.session_state.get("index")

if doc is None or idx is None or not st.session_state.get("index_items"):
    st.warning("まず「① OCR」または「② JSON読み込み」で文書を準備してください。")
else:
    left, right = st.columns([2, 1])
    with left:
        q = st.text_input("検索キーワード", placeholder="例: 材料 / 申請 / 定員 / A5052 / 6061-T6 ...")
    with right:
        st.write(" ")
        clear = st.button("セッションをクリア")
        if clear:
            st.session_state["doc"] = None
            st.session_state["index"] = None
            st.session_state["index_items"] = []
            st.rerun()

    if q.strip():
        results = idx.search(q, top_k=top_k)
        # 足切り
        results = [(it, sc) for it, sc in results if sc >= float(min_score)]

        st.write(f"**ヒット件数（上位表示）:** {len(results)} 件（min_score={min_score}）")

        if not results:
            st.info("該当する結果が見つかりませんでした。PSMを 11/12 に変える、DPIを上げる、二値化/反転を試すのが有効です。")
        else:
            for rank, (it, sc) in enumerate(results, start=1):
                title = f"#{rank}  {it.doc_name} / p.{it.page} / score={sc:.3f}"
                with st.expander(title, expanded=(rank <= 2)):
                    st.markdown(highlight_snippet(it.text, q, max_len=450))

            # 参考：ページ別の文字数サマリ
            pages = doc.get("pages", [])
            df = pd.DataFrame([{"page": p.get("page"), "chars": len((p.get("text") or "").strip())} for p in pages])
            if not df.empty:
                st.caption("ページ別 文字数（OCRで文字がほぼ取れていないページの見つけに有効）")
                st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.info("検索キーワードを入力してください。")

st.caption("※共有（Streamlit Cloud）で使う場合：packages.txt で tesseract/poppler をCloud側に導入してください。ユーザーPCには不要です。")





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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st


# ----------------------------
# 設定
# ----------------------------
OCR_RESULTS_DIR = Path("data") / "ocr_results"

DEFAULT_TOPK = 5

# 容量上限設定（デフォルト値）
# config.pyから読み込もうとしますが、存在しない場合はデフォルト値を使用
MAX_JSON_FILE_SIZE_MB = 100
MAX_TOTAL_CHUNKS = 50000

# config.pyが存在する場合は上書き（オプション）
try:
    import config
    MAX_JSON_FILE_SIZE_MB = getattr(config, 'MAX_JSON_FILE_SIZE_MB', MAX_JSON_FILE_SIZE_MB)
    MAX_TOTAL_CHUNKS = getattr(config, 'MAX_TOTAL_CHUNKS', MAX_TOTAL_CHUNKS)
except (ImportError, AttributeError, Exception):
    # config.pyが存在しない、またはエラーが発生した場合はデフォルト値を使用
    pass


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


