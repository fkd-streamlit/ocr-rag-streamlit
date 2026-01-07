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
    st.warning("`data/ocr_results` に JSON がありません。まず local_ocr_to_json.py でPDF/画像をJSON化して入れてください。")
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


