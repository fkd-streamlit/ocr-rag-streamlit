
# app.py （安定版：Cloudは検索専用）
from __future__ import annotations
import io, json, os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ---- オプション：埋め込みモデル（軽量）とChromaDB ----
EMBED_OK = True
EMBED_ERR = ""
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception as e:
    EMBED_OK = False
    EMBED_ERR = str(e)
    SentenceTransformer = None
    np = None

CHROMA_OK = True
CHROMA_ERR = ""
try:
    import chromadb
except Exception as e:
    CHROMA_OK = False
    CHROMA_ERR = str(e)

# ---- ページ設定 ----
PAGE_TITLE = "技術資料 OCR・RAG 検索（Cloud：検索専用）"
PAGE_ICON = "📄"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# ---- データ構造 ----
@dataclass
class Doc:
    id: str
    title: str
    source: str
    text: str
    uploaded_at: str

# ---- セッション初期化（辞書アクセスで安全に）----
if "docs" not in st.session_state:
    st.session_state["docs"]: List[Doc] = []
if "index" not in st.session_state:
    st.session_state["index"] = None
if "doc_title" not in st.session_state:
    st.session_state["doc_title"] = None

# ---- JSON読み込み ----
def load_json_from_upload(file) -> Optional[Dict]:
    try:
        return json.load(io.BytesIO(file.getvalue()))
    except Exception as e:
        st.error(f"JSON読み込みエラー: {e}")
        return None

def load_json_from_url(url: str) -> Optional[Dict]:
    try:
        import requests
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"URLロード失敗: {e}")
        return None

# ---- スキーマ検証（local_ocr_to_json.py の出力に合わせる）----
def validate_schema(doc: Dict) -> bool:
    # 許容トップキー：doc_id/title/source/created_at/pages
    if not isinstance(doc, dict):
        st.error("JSONのトップが辞書ではありません。"); return False
    for k in ["pages"]:
        if k not in doc:
            st.error(f"キーが不足: {k}"); return False
    if not isinstance(doc["pages"], list) or len(doc["pages"]) == 0:
        st.error("pages が空です。"); return False
    # pages[*].page_num と text を必須化
    for p in doc["pages"]:
        if "text" not in p:
            st.error("pages[*].text がありません。"); return False
        if "page_num" not in p and "page" not in p:
            st.error("pages[*].page_num（または page）がありません。"); return False
    return True

# ---- テキスト結合 ----
def join_text(doc: Dict) -> str:
    buf = []
    for p in doc["pages"]:
        page_no = p.get("page_num", p.get("page", None))
        t = p.get("text", "")
        buf.append(f"=== ページ {page_no} ===\n{t}\n")
    return "\n".join(buf)

# ---- チャンク化 ----
def chunk_text(text: str, chunk_size=1000, overlap=200) -> List[str]:
    chunks = []
    n = len(text)
    i = 0
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks

# ---- 1) TF-IDF インメモリ検索（超軽量／デフォルト）----
class TfIdfIndex:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self.texts: List[str] = []
        self.metas: List[Dict] = []
        self.matrix = None

    def add(self, texts: List[str], metas: List[Dict]):
        self.texts.extend(texts)
        self.metas.extend(metas)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k=5):
        if self.matrix is None or not self.texts:
            return []
        qv = self.vectorizer.transform([query])
        scores = (self.matrix @ qv.T).toarray().ravel()
        idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in idx:
            results.append({
                "score": float(scores[i]),
                "text": self.texts[i],
                "meta": self.metas[i]
            })
        return results

# ---- 2) Sentence-Transformers 検索（軽量埋め込み／任意）----
class EmbeddingIndex:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        if not EMBED_OK:
            raise RuntimeError(f"sentence-transformers の読み込みに失敗: {EMBED_ERR}")
        self.model = SentenceTransformer(model_name)
        self.texts: List[str] = []
        self.metas: List[Dict] = []
        self.embeds = None  # np.ndarray

    def add(self, texts: List[str], metas: List[Dict]):
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if self.embeds is None:
            self.embeds = emb
        else:
            self.embeds = np.vstack([self.embeds, emb])
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query: str, top_k=5):
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = (self.embeds @ q.T).ravel()
        idx = sims.argsort()[::-1][:top_k]
        return [{
            "score": float(sims[i]),
            "text": self.texts[i],
            "meta": self.metas[i]
        } for i in idx]

# ---- インデックス構築（モード選択）----
def build_index(doc: Dict, mode: str = "tfidf", chunk_size=1000, overlap=200):
    texts, metas = [], []
    merged = join_text(doc)
    for c in chunk_text(merged, chunk_size=chunk_size, overlap=overlap):
        texts.append(c)
        metas.append({
            "page_num": None,
            "title": doc.get("title") or doc.get("source") or "",
            "source": doc.get("source") or doc.get("doc_id") or "",
        })
    if mode == "tfidf":
        idx = TfIdfIndex()
        idx.add(texts, metas)
        return idx
    elif mode == "embed":
        idx = EmbeddingIndex(model_name=st.session_state.get("embed_model", "paraphrase-multilingual-MiniLM-L12-v2"))
        idx.add(texts, metas)
        return idx
    else:
        raise ValueError("mode は 'tfidf' か 'embed' を指定してください。")

# ---- サイドバー ----
with st.sidebar:
    st.header("⚙️ Cloud運用モード")
    mode = st.radio("検索モード", ["TF‑IDF（軽量・推奨）", "埋め込み（Sentence‑Transformers）"], index=0)
    top_k = st.slider("Top‑K", 1, 10, 5)
    st.caption("※ 埋め込みモードは初回ダウンロードに時間がかかる場合があります。")
    if mode.startswith("埋め込み"):
        st.session_state["embed_model"] = st.selectbox(
            "モデル", ["paraphrase-multilingual-MiniLM-L12-v2", "all-MiniLM-L6-v2"], index=0
        )
    st.markdown("---")
    st.subheader("RAG（ChromaDB）")
    if CHROMA_OK:
        st.caption("※ 今回はインメモリ検索のみ（ChromaDBは未使用）。必要なら後で追加可能。")
    else:
        st.warning(f"ChromaDB 未使用（理由: {CHROMA_ERR}）")

# ---- メインUI ----
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
tab1, tab2, tab3 = st.tabs(["📤 JSONアップロード", "🔗 URLからロード", "🔍 検索"])

# 1) JSONアップロード
with tab1:
    st.header("📤 ローカルOCR生成の JSON を読み込む")
    up = st.file_uploader("OCR結果JSON（local_ocr_to_json.py の出力）", type=["json"])
    if up:
        doc = load_json_from_upload(up)
        if doc and validate_schema(doc):
            idx_mode = "tfidf" if mode.startswith("TF") else "embed"
            with st.spinner("インデックス構築中…"):
                st.session_state["index"] = build_index(doc, mode=idx_mode, chunk_size=1000, overlap=200)
                st.session_state["doc_title"] = doc.get("title") or doc.get("source") or up.name
            st.success("読み込み完了。検索タブへどうぞ。")

# 2) URLロード
with tab2:
    st.header("🔗 GitHub Raw 等の URL から読み込み")
    url = st.text_input("JSONのURL")
    if st.button("URLから読み込む") and url.strip():
        doc = load_json_from_url(url.strip())
        if doc and validate_schema(doc):
            idx_mode = "tfidf" if mode.startswith("TF") else "embed"
            with st.spinner("インデックス構築中…"):
                st.session_state["index"] = build_index(doc, mode=idx_mode, chunk_size=1000, overlap=200)
                st.session_state["doc_title"] = doc.get("title") or doc.get("source") or url
            st.success("読み込み完了。検索タブへどうぞ。")

# 3) 検索
with tab3:
    st.header(f"🔍 検索（{st.session_state.get('doc_title') or '未読込'}）")
    if st.session_state.get("index") is None:
        st.info("まず JSON を読み込んでください。")
    else:
        q = st.text_input("質問・キーワード")
        if st.button("検索") and q.strip():
            try:
                results = st.session_state["index"].search(q.strip(), top_k=top_k)
                if results:
                    st.subheader(f"検索結果：{len(results)}件")
                    for i, r in enumerate(results, 1):
                        score = r.get("score")
                        with st.expander(f"結果 {i}（score≈{score:.3f}）"):
                            st.write("**メタ**")
                            st.json(r.get("meta", {}))
                            st.write("**該当テキスト（chunk）**")
                            st.text(r["text"])
                else:
                    st.info("該当なし。キーワードや表記揺れを調整してください。")
            except Exception as e:
                st.error(f"検索中にエラー: {e}")


