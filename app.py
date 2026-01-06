"""
技術資料RAG検索アプリ（Cloud検索専用版 / ルートC）
- ローカルでOCR→JSON化した結果（data/ocr_results/*.json）を読み込み
- ChromaDB + SentenceTransformers で検索（RAGの「R」部分）
- Streamlit CloudではOCRを一切しない（Tesseract不要）
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st

# -----------------------------
# 設定読み込み（config.pyがあれば使う）
# -----------------------------
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

# ディレクトリ確保
OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# RAGライブラリ（必須）
# -----------------------------
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    CHROMA_OK = True
except Exception:
    CHROMA_OK = False

# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="技術資料RAG検索（Cloud検索専用）",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔎 技術資料RAG検索（Cloud検索専用 / ルートC）")
st.caption("※ このCloudアプリはOCRしません。ローカルで作ったJSONを読み込んで検索します。")
st.markdown("---")


# -----------------------------
# ユーティリティ
# -----------------------------
def load_json_documents(json_dir: Path) -> List[Dict[str, Any]]:
    """data/ocr_results/*.json を読み込む"""
    docs: List[Dict[str, Any]] = []
    for p in sorted(json_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            # 必須キーを正規化
            doc_id = obj.get("id") or p.stem
            filename = obj.get("filename") or obj.get("source") or p.name
            text = obj.get("text") or ""
            uploaded_at = obj.get("uploaded_at") or obj.get("created_at") or ""
            ocr_settings = obj.get("ocr_settings") or {}

            docs.append(
                {
                    "id": str(doc_id),
                    "filename": str(filename),
                    "text": str(text),
                    "uploaded_at": str(uploaded_at),
                    "ocr_settings": ocr_settings,
                    "_path": str(p),
                }
            )
        except Exception:
            # 壊れたJSONが混ざっていても落とさない
            continue
    return docs


@st.cache_resource
def get_embedding_model(model_name: str):
    return SentenceTransformer(model_name)


@st.cache_resource
def get_chroma_collection(persist_dir: str, collection_name: str):
    """
    Chroma の永続DBを開く（バージョン差分に強めに）
    """
    # 新しめのAPI: PersistentClient
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        try:
            col = client.get_collection(collection_name)
        except Exception:
            col = client.create_collection(collection_name)
        return col
    except Exception:
        # 古いAPI: Client + Settings
        try:
            from chromadb.config import Settings

            client = chromadb.Client(
                Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
            )
            try:
                col = client.get_collection(collection_name)
            except Exception:
                col = client.create_collection(collection_name)
            return col
        except Exception as e:
            raise RuntimeError(f"Chroma初期化に失敗: {e}") from e


def ensure_indexed(
    docs: List[Dict[str, Any]],
    collection,
    model,
) -> Dict[str, Any]:
    """
    JSONドキュメントをChromaに投入（未登録のみ）
    """
    # 既存ID一覧を取得（大量だと重いので、まずは docs のidだけ確認）
    # Chromaに "get(ids=[...])" が通れば、それを使う
    to_add = []
    to_add_ids = []
    to_add_texts = []
    to_add_metas = []

    # docごとに存在確認
    for d in docs:
        doc_id = d["id"]
        exists = False
        try:
            got = collection.get(ids=[doc_id])
            if got and got.get("ids") and len(got["ids"]) > 0:
                exists = True
        except Exception:
            # get(ids=) が失敗する実装もあるので、その場合は追加側で弾かれる想定
            exists = False

        if not exists:
            text = (d.get("text") or "").strip()
            if not text:
                continue
            to_add.append(d)
            to_add_ids.append(doc_id)
            to_add_texts.append(text)
            to_add_metas.append(
                {
                    "filename": d.get("filename", ""),
                    "uploaded_at": d.get("uploaded_at", ""),
                    "json_path": d.get("_path", ""),
                }
            )

    if not to_add_ids:
        return {"added": 0}

    embeddings = model.encode(to_add_texts).tolist()
    collection.add(
        ids=to_add_ids,
        embeddings=embeddings,
        documents=to_add_texts,
        metadatas=to_add_metas,
    )
    return {"added": len(to_add_ids)}


def search(query: str, n_results: int, collection, model) -> List[Dict[str, Any]]:
    q_emb = model.encode([query]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=n_results)

    out: List[Dict[str, Any]] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]  # 小さいほど近い

    for i in range(len(ids)):
        out.append(
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return out


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Cloud検索専用（ルートC）")
    st.write("✅ OCRはローカルで実施し、JSONをこのリポジトリへ配置します。")
    st.write(f"📁 JSON読込先: `{OCR_RESULTS_DIR.as_posix()}`")
    st.write(f"🧠 埋め込みモデル: `{EMBEDDING_MODEL_NAME}`")
    st.write(f"🗄️ ChromaDB: `{VECTOR_DB_DIR.as_posix()}`")

    st.markdown("---")

    if not CHROMA_OK:
        st.error("ChromaDB / sentence-transformers がrequirementsに入っていません。")
        st.stop()

    if st.button("🔄 JSONを再読み込み＆再インデックス"):
        # cacheを使っていても docs は毎回読むが、ユーザーに明示したいので rerun
        st.session_state["_force_reindex"] = True
        st.rerun()

    st.markdown("---")
    st.caption("※ Streamlit CloudのファイルはGitHubに置いたものが読まれます。")


# -----------------------------
# メイン処理
# -----------------------------
docs = load_json_documents(OCR_RESULTS_DIR)

tab_search, tab_docs, tab_status = st.tabs(["🔍 検索", "📚 JSON一覧", "🧪 状態/診断"])

with tab_status:
    st.subheader("状態")
    st.write(f"JSON件数: **{len(docs)}**")
    if len(docs) == 0:
        st.warning(
            "まだJSONがありません。ローカルで `local_ocr_to_json.py` を実行してJSONを作成し、"
            "`data/ocr_results/` に入れてGitHubへpushしてください。"
        )

    st.write("Chroma/Model 初期化…")
    try:
        collection = get_chroma_collection(str(VECTOR_DB_DIR), VECTOR_DB_COLLECTION_NAME)
        model = get_embedding_model(EMBEDDING_MODEL_NAME)
        st.success("✅ ChromaDBと埋め込みモデルの初期化OK")
    except Exception as e:
        st.error(f"初期化失敗: {e}")
        st.stop()

    # 自動インデックス
    if len(docs) > 0:
        try:
            force = st.session_state.pop("_force_reindex", False)
            if force:
                # force時は既存を消したいケースもあるが、ここでは追加のみ（安全）
                st.info("再インデックス（追加）を実行します…")
            result = ensure_indexed(docs, collection, model)
            st.write(f"今回追加した件数: **{result.get('added', 0)}**")
        except Exception as e:
            st.error(f"インデックス失敗: {e}")

    st.markdown("---")
    st.write("📌 ルートCではCloud側にTesseractは不要です（OCRはしません）。")


with tab_docs:
    st.subheader("JSON一覧（data/ocr_results）")
    if len(docs) == 0:
        st.info("JSONがありません。まずローカルでJSON生成→GitHubへpushしてください。")
    else:
        for d in docs:
            with st.expander(f"📄 {d.get('filename','')}  |  {d.get('id','')}"):
                st.write(f"JSON: `{d.get('_path','')}`")
                if d.get("uploaded_at"):
                    st.write(f"日時: {d.get('uploaded_at')}")
                st.write("テキスト（先頭500文字）:")
                t = (d.get("text") or "")
                st.text(t[:500] + ("..." if len(t) > 500 else ""))


with tab_search:
    st.subheader("🔍 検索（RAGのRetrieval）")

    if not CHROMA_OK:
        st.stop()

    if len(docs) == 0:
        st.info("まずJSONを `data/ocr_results/` に入れてGitHubへpushしてください。")
        st.stop()

    # 初期化
    collection = get_chroma_collection(str(VECTOR_DB_DIR), VECTOR_DB_COLLECTION_NAME)
    model = get_embedding_model(EMBEDDING_MODEL_NAME)

    # 自動で未登録分を追加
    try:
        ensure_indexed(docs, collection, model)
    except Exception as e:
        st.error(f"インデックス処理でエラー: {e}")
        st.stop()

    query = st.text_input("検索クエリ", placeholder="例：工程異常の原因、材料の特性、設備点検…")
    n_results = st.slider("検索結果数", 1, int(MAX_SEARCH_RESULTS), int(DEFAULT_SEARCH_RESULTS))

    if st.button("🔎 検索", type="primary") and query.strip():
        with st.spinner("検索中…"):
            results = search(query.strip(), n_results, collection, model)

        if not results:
            st.info("検索結果が見つかりませんでした。")
        else:
            st.success(f"検索結果: {len(results)}件")
            for i, r in enumerate(results, 1):
                dist = r.get("distance")
                score = None if dist is None else (1.0 / (1.0 + float(dist)))  # ざっくり表示
                title = f"{i}. {r.get('id','')}"
                if score is not None:
                    title += f"  |  近さ目安: {score:.3f}"

                with st.expander(title):
                    st.write("メタデータ")
                    st.json(r.get("metadata") or {})
                    st.write("本文（先頭800文字）")
                    txt = r.get("text") or ""
                    st.text(txt[:800] + ("..." if len(txt) > 800 else ""))

