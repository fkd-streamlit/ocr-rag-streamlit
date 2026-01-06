# app.py
"""
技術資料OCR・RAG検索アプリケーション（ルートC：OCRはローカル、Cloudは検索共有）
- ローカルで作成したOCR結果JSONをアップロードして蓄積
- ChromaDB + SentenceTransformers があればRAG検索（ベクトル検索）
- ない場合でも簡易検索（部分一致）で最低限動作
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# -------------------------
# 0) 設定（config.py があれば優先）
# -------------------------
try:
    from config import (
        DATA_DIR,
        OCR_RESULTS_DIR,
        VECTOR_DB_DIR,
        VECTOR_DB_COLLECTION_NAME,
        EMBEDDING_MODEL_NAME,
        DEFAULT_SEARCH_RESULTS,
        MAX_SEARCH_RESULTS,
    )
except Exception:
    DATA_DIR = Path("data")
    OCR_RESULTS_DIR = DATA_DIR / "ocr_results"
    VECTOR_DB_DIR = DATA_DIR / "chroma_db"
    VECTOR_DB_COLLECTION_NAME = "technical_documents"
    EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    DEFAULT_SEARCH_RESULTS = 5
    MAX_SEARCH_RESULTS = 10

for d in [DATA_DIR, OCR_RESULTS_DIR, VECTOR_DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1) RAG依存の読み込み（あれば使う）
# -------------------------
CHROMADB_AVAILABLE = True
CHROMA_IMPORT_ERROR = ""

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except Exception as e:
    CHROMADB_AVAILABLE = False
    CHROMA_IMPORT_ERROR = str(e)

# -------------------------
# 2) UI設定
# -------------------------
st.set_page_config(
    page_title="技術資料OCR・RAG検索（ルートC）",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# 3) データ構造
# -------------------------
REQUIRED_JSON_KEYS = {"id", "filename", "text", "uploaded_at"}

@dataclass
class Doc:
    id: str
    filename: str
    text: str
    uploaded_at: str
    meta: Dict[str, Any]

def now_id(prefix: str = "doc") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def safe_load_json(b: bytes) -> Dict[str, Any]:
    return json.loads(b.decode("utf-8"))

def validate_doc_json(obj: Dict[str, Any]) -> Tuple[bool, str]:
    missing = REQUIRED_JSON_KEYS - set(obj.keys())
    if missing:
        return False, f"必須キーが不足しています: {sorted(list(missing))}"
    if not isinstance(obj.get("text", ""), str):
        return False, "text は文字列である必要があります"
    if not obj["id"]:
        return False, "id が空です"
    return True, ""

def normalize_doc(obj: Dict[str, Any]) -> Doc:
    meta = dict(obj)
    return Doc(
        id=str(obj["id"]),
        filename=str(obj.get("filename", "")),
        text=str(obj.get("text", "")),
        uploaded_at=str(obj.get("uploaded_at", "")),
        meta=meta,
    )

# -------------------------
# 4) JSON保管（Cloudでは永続保証なしだが、共有用途は「アップロード」運用でOK）
# -------------------------
def list_saved_json_files() -> List[Path]:
    return sorted(OCR_RESULTS_DIR.glob("*.json"))

def load_docs_from_disk() -> List[Doc]:
    docs: List[Doc] = []
    for p in list_saved_json_files():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            ok, msg = validate_doc_json(obj)
            if not ok:
                continue
            docs.append(normalize_doc(obj))
        except Exception:
            continue
    return docs

def save_doc_to_disk(doc: Doc) -> Path:
    out = OCR_RESULTS_DIR / f"{doc.id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc.meta, f, ensure_ascii=False, indent=2)
    return out

def delete_doc_files(doc_id: str) -> None:
    p = OCR_RESULTS_DIR / f"{doc_id}.json"
    if p.exists():
        p.unlink()

# -------------------------
# 5) テキスト分割（RAG用 chunking）
# -------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    シンプルな文字数ベース分割（日本語向けに安定）
    """
    text = text.replace("\r\n", "\n")
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
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

# -------------------------
# 6) RAG（ChromaDB）
# -------------------------
@st.cache_resource
def get_embedding_model() -> Optional["SentenceTransformer"]:
    if not CHROMADB_AVAILABLE:
        return None
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_chroma_collection():
    if not CHROMADB_AVAILABLE:
        return None
    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(VECTOR_DB_DIR),
        )
    )
    try:
        col = client.get_collection(VECTOR_DB_COLLECTION_NAME)
    except Exception:
        col = client.create_collection(VECTOR_DB_COLLECTION_NAME)
    return col

def rag_add_doc(doc: Doc) -> Tuple[bool, str]:
    """
    1文書を chunk に分解してベクトルDBに追加
    """
    if not CHROMADB_AVAILABLE:
        return False, "ChromaDB/SentenceTransformers が未導入です"

    model = get_embedding_model()
    col = get_chroma_collection()
    if model is None or col is None:
        return False, "RAG初期化に失敗しました"

    chunks = chunk_text(doc.text)
    ids = [f"{doc.id}__c{i:04d}" for i in range(len(chunks))]
    metadatas = []
    for i in range(len(chunks)):
        metadatas.append(
            {
                "doc_id": doc.id,
                "chunk_index": i,
                "filename": doc.filename,
                "uploaded_at": doc.uploaded_at,
            }
        )

    try:
        emb = model.encode(chunks).tolist()
        col.add(ids=ids, embeddings=emb, documents=chunks, metadatas=metadatas)
        return True, f"RAGに登録しました（chunks={len(chunks)}）"
    except Exception as e:
        return False, f"RAG登録エラー: {e}"

def rag_delete_doc(doc_id: str) -> Tuple[bool, str]:
    if not CHROMADB_AVAILABLE:
        return False, "ChromaDB未導入"
    col = get_chroma_collection()
    if col is None:
        return False, "コレクション取得失敗"

    try:
        # doc_id__c0000 のようなIDをまとめて削除
        # where で doc_id 指定できればベストだが、環境差があるので get→filter で対応
        # 取得件数が多くなる場合は運用で分割してください
        all_ids = col.get(include=["metadatas"]).get("ids", [])
        all_metas = col.get(include=["metadatas"]).get("metadatas", [])
        del_ids = []
        for _id, m in zip(all_ids, all_metas):
            if isinstance(m, dict) and m.get("doc_id") == doc_id:
                del_ids.append(_id)
        if del_ids:
            col.delete(ids=del_ids)
        return True, f"RAGから削除しました（{len(del_ids)}件）"
    except Exception as e:
        return False, f"RAG削除エラー: {e}"

def rag_search(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    if not CHROMADB_AVAILABLE:
        return []

    model = get_embedding_model()
    col = get_chroma_collection()
    if model is None or col is None:
        return []

    try:
        qemb = model.encode([query]).tolist()
        res = col.query(query_embeddings=qemb, n_results=n_results, include=["documents", "metadatas", "distances", "ids"])
        out: List[Dict[str, Any]] = []
        if res and res.get("ids") and len(res["ids"][0]) > 0:
            for i in range(len(res["ids"][0])):
                out.append(
                    {
                        "id": res["ids"][0][i],
                        "text": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i],
                        "distance": res["distances"][0][i] if res.get("distances") else None,
                    }
                )
        return out
    except Exception:
        return []

# -------------------------
# 7) 簡易検索（RAGが無いとき）
# -------------------------
def simple_search(docs: List[Doc], query: str, limit: int = 5) -> List[Dict[str, Any]]:
    q = query.strip()
    if not q:
        return []
    hits = []
    for d in docs:
        idx = d.text.find(q)
        if idx >= 0:
            start = max(0, idx - 120)
            end = min(len(d.text), idx + 400)
            snippet = d.text[start:end]
            hits.append(
                {
                    "doc_id": d.id,
                    "filename": d.filename,
                    "snippet": snippet,
                    "pos": idx,
                }
            )
    hits.sort(key=lambda x: x["pos"])
    return hits[:limit]

# -------------------------
# 8) セッション初期化
# -------------------------
if "documents" not in st.session_state:
    st.session_state.documents: List[Doc] = load_docs_from_disk()

# -------------------------
# 9) サイドバー
# -------------------------
with st.sidebar:
    st.header("⚙️ 共有・検索モード（ルートC）")

    st.markdown("**このアプリはCloud上でOCRしません。** 代わりにローカルで作ったOCR結果JSONを取り込みます。")

    st.markdown("---")
    st.subheader("RAG状態")
    if CHROMADB_AVAILABLE:
        st.success("✅ RAG（ChromaDB + Embedding）利用可能")
        st.caption(f"Embedding: {EMBEDDING_MODEL_NAME}")
    else:
        st.warning("⚠️ RAGは未使用（簡易検索で動作）")
        st.caption(f"理由: {CHROMA_IMPORT_ERROR}")

    st.markdown("---")
    st.subheader("保存データ")
    st.write(f"JSON保存先: `{OCR_RESULTS_DIR.as_posix()}`")
    st.write(f"保存済み文書数: **{len(st.session_state.documents)}**")

    if st.button("🔄 ディスクから再読み込み"):
        st.session_state.documents = load_docs_from_disk()
        st.success("再読み込みしました")
        st.rerun()

# -------------------------
# 10) メインUI
# -------------------------
st.title("📄 技術資料OCR・RAG検索（ルートC：JSON取り込み → 共有検索）")
st.markdown("---")

tab_upload, tab_search, tab_list = st.tabs(["📤 JSON取り込み", "🔍 検索", "📚 文書一覧"])

# ========== Tab 1: JSON取り込み ==========
with tab_upload:
    st.header("📤 OCR結果JSONの取り込み（ローカルで作成したもの）")

    st.markdown(
        """
- ローカルOCRで作った **JSON（1文書=1ファイル）** をアップロードしてください  
- アップロード後、（RAGが有効なら）ベクトルDBにも登録できます
"""
    )

    st.subheader("JSONアップロード")
    up = st.file_uploader("OCR結果JSON（.json）を選択", type=["json"], accept_multiple_files=True)

    colA, colB = st.columns([1, 1])
    with colA:
        add_to_rag = st.checkbox("RAGにも登録する（おすすめ）", value=CHROMADB_AVAILABLE, disabled=not CHROMADB_AVAILABLE)
    with colB:
        save_disk = st.checkbox("サーバ側にJSONとして保存する", value=True, help="Cloudでは永続保証はありません（運用はアップロード推奨）")

    if up:
        for f in up:
            try:
                obj = safe_load_json(f.getvalue())
                ok, msg = validate_doc_json(obj)
                if not ok:
                    st.error(f"❌ {f.name}: {msg}")
                    continue

                doc = normalize_doc(obj)

                # 同IDが既にある場合はスキップ（上書きしたいならIDを変える運用）
                if any(d.id == doc.id for d in st.session_state.documents):
                    st.warning(f"⚠️ {doc.id} は既に登録済みです（スキップ）")
                    continue

                st.session_state.documents.append(doc)

                if save_disk:
                    save_doc_to_disk(doc)

                if add_to_rag and CHROMADB_AVAILABLE:
                    ok2, msg2 = rag_add_doc(doc)
                    if ok2:
                        st.success(f"✅ {doc.filename}: {msg2}")
                    else:
                        st.warning(f"⚠️ {doc.filename}: {msg2}")
                else:
                    st.success(f"✅ {doc.filename}: 取り込みました")

            except Exception as e:
                st.error(f"❌ {f.name}: 読み込み失敗: {e}")

        st.info("取り込み後は「検索」タブで検索できます。")

    st.markdown("---")
    st.subheader("JSONフォーマット例（参考）")
    example = {
        "id": "doc_20260106_120000",
        "filename": "技術資料A.pdf",
        "text": "（OCRで抽出した本文テキスト…）",
        "uploaded_at": datetime.now().isoformat(),
        "ocr_settings": {
            "source": "local_ocr",
            "lang": "jpn",
            "psm": 6,
            "oem": 3
        }
    }
    st.code(json.dumps(example, ensure_ascii=False, indent=2), language="json")

# ========== Tab 2: 検索 ==========
with tab_search:
    st.header("🔍 検索（RAGまたは簡易検索）")

    if len(st.session_state.documents) == 0:
        st.info("📝 まず「JSON取り込み」タブからOCR結果JSONを取り込んでください。")
    else:
        query = st.text_input("検索クエリ", placeholder="例：金型温度の設定、材料の強度、サーボ調整…")
        n_results = st.slider("検索結果数", 1, MAX_SEARCH_RESULTS, DEFAULT_SEARCH_RESULTS)

        if st.button("🔍 検索実行", type="primary") and query.strip():
            if CHROMADB_AVAILABLE:
                with st.spinner("RAG検索中..."):
                    results = rag_search(query, n_results=n_results)

                if results:
                    st.subheader(f"検索結果（RAG）: {len(results)}件")
                    for i, r in enumerate(results, 1):
                        meta = r.get("metadata") or {}
                        dist = r.get("distance")
                        score = None
                        if isinstance(dist, (int, float)):
                            # Chromaは距離が小さいほど近い。見た目用にスコア化
                            score = 1.0 / (1.0 + dist)

                        title = f"結果 {i}: {meta.get('filename','(unknown)')} / doc={meta.get('doc_id','')}"
                        if score is not None:
                            title += f" / score≈{score:.3f}"

                        with st.expander(title):
                            st.write("**メタデータ**")
                            st.json(meta)
                            st.write("**該当テキスト（chunk）**")
                            st.text(r.get("text", ""))
                else:
                    st.info("検索結果が見つかりませんでした（RAG）")
                    st.caption("※ 取り込み直後は、RAG登録に失敗している可能性があります。JSON取り込みタブで「RAGにも登録」にチェックして再取り込みしてください。")
            else:
                with st.spinner("簡易検索中..."):
                    hits = simple_search(st.session_state.documents, query, limit=n_results)

                if hits:
                    st.subheader(f"検索結果（簡易）: {len(hits)}件")
                    for i, h in enumerate(hits, 1):
                        with st.expander(f"結果 {i}: {h['filename']} / doc={h['doc_id']}"):
                            st.text(h["snippet"])
                else:
                    st.info("検索結果が見つかりませんでした（簡易）")

# ========== Tab 3: 文書一覧 ==========
with tab_list:
    st.header("📚 保存済み文書一覧")

    if len(st.session_state.documents) == 0:
        st.info("📝 文書がありません。JSON取り込みを行ってください。")
    else:
        st.write(f"**文書数: {len(st.session_state.documents)}**")

        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc.filename}（{doc.id}）"):
                st.write(f"**uploaded_at:** {doc.uploaded_at}")
                st.write(f"**文字数:** {len(doc.text)}")

                # プレビュー
                preview = doc.text[:800] + ("..." if len(doc.text) > 800 else "")
                st.text(preview)

                c1, c2, c3 = st.columns([1, 1, 2])

                with c1:
                    # JSONダウンロード
                    st.download_button(
                        "⬇️ JSONをダウンロード",
                        data=json.dumps(doc.meta, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"{doc.id}.json",
                        mime="application/json",
                        key=f"dl_{doc.id}",
                    )

                with c2:
                    if CHROMADB_AVAILABLE:
                        if st.button("🧠 RAGに登録", key=f"rag_add_{doc.id}"):
                            ok, msg = rag_add_doc(doc)
                            (st.success if ok else st.warning)(msg)

                with c3:
                    if st.button("🗑️ 削除（ローカル保存分も）", key=f"del_{doc.id}"):
                        # セッションから削除
                        st.session_state.documents = [d for d in st.session_state.documents if d.id != doc.id]

                        # ディスクJSON削除
                        delete_doc_files(doc.id)

                        # RAG削除
                        if CHROMADB_AVAILABLE:
                            rag_delete_doc(doc.id)

                        st.success("削除しました")
                        st.rerun()



