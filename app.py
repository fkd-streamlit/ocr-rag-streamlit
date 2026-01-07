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




