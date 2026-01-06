"""
技術資料OCR・RAG検索アプリケーション
PDF/画像をOCRで読み込み、RAGで検索可能にするWebアプリ
"""

import streamlit as st
import pdf2image
from PIL import Image, ImageEnhance
import numpy as np
import pytesseract
import os
import platform
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
import tempfile

# ------------------------------------------------------------
# 0) OpenCV (cv2) は Streamlit Cloud で libGL.so.1 問題が出やすいので安全に扱う
# ------------------------------------------------------------
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = ""
except Exception as e:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(e)

# ------------------------------------------------------------
# 1) ページ設定（Streamlitの最初のUI呼び出しにする必要あり）
# ------------------------------------------------------------
st.set_page_config(
    page_title="技術資料OCR・RAG検索",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 2) 設定の読み込み（config.py があればそれを使う）
# ------------------------------------------------------------
try:
    from config import (
        DATA_DIR, UPLOADS_DIR, OCR_RESULTS_DIR, VECTOR_DB_DIR,
        TESSERACT_LANG, TESSERACT_PSM_DEFAULT, TESSERACT_OEM_DEFAULT,
        PDF_DPI, DEFAULT_CONTRAST, DEFAULT_BRIGHTNESS, DEFAULT_THRESHOLD,
        DEFAULT_USE_ADAPTIVE, VECTOR_DB_COLLECTION_NAME, EMBEDDING_MODEL_NAME,
        DEFAULT_SEARCH_RESULTS, MAX_SEARCH_RESULTS
    )
except Exception:
    DATA_DIR = Path("data")
    UPLOADS_DIR = DATA_DIR / "uploads"
    OCR_RESULTS_DIR = DATA_DIR / "ocr_results"
    VECTOR_DB_DIR = DATA_DIR / "chroma_db"
    TESSERACT_LANG = "jpn"
    TESSERACT_PSM_DEFAULT = 6
    TESSERACT_OEM_DEFAULT = 3
    PDF_DPI = 300
    DEFAULT_CONTRAST = 1.0
    DEFAULT_BRIGHTNESS = 0
    DEFAULT_THRESHOLD = 127
    DEFAULT_USE_ADAPTIVE = False
    VECTOR_DB_COLLECTION_NAME = "technical_documents"
    EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    DEFAULT_SEARCH_RESULTS = 5
    MAX_SEARCH_RESULTS = 10

# ディレクトリ作成（存在しない場合）
for dir_path in [DATA_DIR, UPLOADS_DIR, OCR_RESULTS_DIR, VECTOR_DB_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 3) Windows環境でのTesseract OCRパス設定（Cloud/Linuxでは不要）
# ------------------------------------------------------------
TESSERACT_WARNING = None
if platform.system() == "Windows":
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if not tesseract_cmd:
        for path in tesseract_paths:
            if os.path.exists(path):
                tesseract_cmd = path
                break

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        TESSERACT_WARNING = (
            "⚠️ Tesseract OCRが見つかりません。\n\n"
            "インストールパス: `C:\\Program Files\\Tesseract-OCR`\n\n"
            "環境変数 `TESSERACT_CMD` を設定するか、パス設定を確認してください。"
        )

# ------------------------------------------------------------
# 4) RAG関連のインポート（Streamlit Cloudで未導入の可能性があるため安全に）
#    ※ ここでは st.warning しない（page_config後ならOKだが、UI汚染回避）
# ------------------------------------------------------------
CHROMADB_AVAILABLE = False
CHROMA_IMPORT_ERROR = ""
try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    CHROMADB_AVAILABLE = True
except Exception as e:
    CHROMADB_AVAILABLE = False
    CHROMA_IMPORT_ERROR = str(e)

# ------------------------------------------------------------
# 5) セッション状態の初期化
# ------------------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = []
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

# ------------------------------------------------------------
# 6) 起動時に保存済みJSONを読み込む（Cloudでも“起動中は”使える）
# ------------------------------------------------------------
def load_saved_documents() -> List[Dict[str, Any]]:
    docs = []
    try:
        for p in sorted(OCR_RESULTS_DIR.glob("doc_*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    docs.append(json.load(f))
            except Exception:
                continue
    except Exception:
        pass
    return docs

if len(st.session_state.documents) == 0:
    st.session_state.documents = load_saved_documents()

# ------------------------------------------------------------
# 7) 画像前処理（cv2あり→高機能 / cv2なし→PILでフォールバック）
# ------------------------------------------------------------
def preprocess_image(
    image: Image.Image,
    contrast: float = 1.0,
    brightness: float = 0,
    threshold: int = 127,
    use_adaptive: bool = False,
) -> Image.Image:
    """
    画像の前処理（コントラスト、明度、閾値処理）
    - OpenCVが使える場合：cv2でグレースケール/適応閾値
    - OpenCVが使えない場合：PILでコントラスト/明度/単純2値化
    """
    if CV2_AVAILABLE and cv2 is not None:
        # PIL → OpenCV形式
        cv_img = np.array(image.convert("RGB"))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        # グレースケール
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # コントラスト・明度
        if contrast != 1.0 or brightness != 0:
            gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

        # 閾値
        if use_adaptive:
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        else:
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # OpenCV → PIL
        result = Image.fromarray(thresh).convert("RGB")
        return result

    # ---- PILフォールバック（Cloudで確実に動く） ----
    img = image.convert("L")  # grayscale

    # コントラスト
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    # 明度（-100..100 を想定、PILは倍率なので近似）
    if brightness != 0:
        # brightness: -100..100 -> factor 0.5..1.5 程度にマップ
        factor = max(0.1, 1.0 + (brightness / 200.0))
        img = ImageEnhance.Brightness(img).enhance(factor)

    # 適応閾値はOpenCVなしでは不可 → 単純2値化
    thr = int(np.clip(threshold, 0, 255))
    img = img.point(lambda p: 255 if p > thr else 0)
    return img.convert("RGB")


# ------------------------------------------------------------
# 8) OCR
# ------------------------------------------------------------
def perform_ocr(
    image: Image.Image,
    lang: str = "jpn",
    psm: int = 6,
    oem: int = 3,
) -> Dict[str, Any]:
    """OCR処理を実行"""
    try:
        custom_config = f"--oem {oem} --psm {psm} -l {lang}"
        text = pytesseract.image_to_string(image, config=custom_config)
        data = pytesseract.image_to_data(
            image, config=custom_config, output_type=pytesseract.Output.DICT
        )
        return {
            "text": text,
            "data": data,
            "word_count": len([w for w in text.split() if w.strip()]),
            "char_count": len(text),
        }
    except Exception as e:
        st.error(f"OCRエラー: {str(e)}")
        return {"text": "", "data": {}, "word_count": 0, "char_count": 0}


# ------------------------------------------------------------
# 9) RAG (ChromaDB)
# ------------------------------------------------------------
def initialize_vector_db():
    """ベクトルDBを初期化"""
    if not CHROMADB_AVAILABLE:
        return None
    try:
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(VECTOR_DB_DIR)
        ))
        try:
            collection = client.get_collection(VECTOR_DB_COLLECTION_NAME)
        except Exception:
            collection = client.create_collection(VECTOR_DB_COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"ベクトルDB初期化エラー: {str(e)}")
        return None


def load_embedding_model():
    """埋め込みモデルを読み込み（インデントバグ修正版）"""
    if not CHROMADB_AVAILABLE:
        return None
    try:
        if st.session_state.embedding_model is None:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            st.session_state.embedding_model = model
        return st.session_state.embedding_model
    except Exception as e:
        st.error(f"埋め込みモデル読み込みエラー: {str(e)}")
        return None


def save_document_to_vector_db(doc_id: str, text: str, metadata: Dict[str, Any]):
    """文書をベクトルDBに保存"""
    if not CHROMADB_AVAILABLE:
        return False
    try:
        collection = initialize_vector_db()
        model = load_embedding_model()
        if collection is None or model is None:
            return False

        embeddings = model.encode([text]).tolist()
        collection.add(
            ids=[doc_id],
            embeddings=embeddings,
            documents=[text],
            metadatas=[metadata],
        )
        return True
    except Exception as e:
        st.error(f"ベクトルDB保存エラー: {str(e)}")
        return False


def search_vector_db(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """ベクトルDBから検索"""
    if not CHROMADB_AVAILABLE:
        return []
    try:
        collection = initialize_vector_db()
        model = load_embedding_model()
        if collection is None or model is None:
            return []

        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=n_results)

        search_results = []
        if results.get("ids") and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                dist = None
                if "distances" in results and results["distances"]:
                    dist = results["distances"][0][i]
                search_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": dist,
                })
        return search_results
    except Exception as e:
        st.error(f"検索エラー: {str(e)}")
        return []


# ------------------------------------------------------------
# 10) メイン
# ------------------------------------------------------------
def main():
    # タイトル
    st.title("📄 技術資料OCR・RAG検索システム")
    st.markdown("---")

    # サイドバー: OCR設定
    with st.sidebar:
        st.header("⚙️ OCR設定")

        if TESSERACT_WARNING:
            st.warning(TESSERACT_WARNING)

        # OpenCV状況表示
        if CV2_AVAILABLE:
            st.success("✅ OpenCV (cv2) 利用可能")
        else:
            st.warning("⚠️ OpenCV (cv2) が利用できません（Cloudではよくあります）")
            st.caption(f"cv2 error: {CV2_IMPORT_ERROR}")

        st.subheader("画像前処理")
        contrast = st.slider(
            "コントラスト",
            min_value=0.5,
            max_value=2.0,
            value=float(DEFAULT_CONTRAST),
            step=0.1,
            help="画像のコントラストを調整します",
        )

        brightness = st.slider(
            "明度",
            min_value=-100,
            max_value=100,
            value=int(DEFAULT_BRIGHTNESS),
            step=10,
            help="画像の明るさを調整します",
        )

        use_adaptive = st.checkbox(
            "適応的閾値処理を使用",
            value=bool(DEFAULT_USE_ADAPTIVE) and CV2_AVAILABLE,
            disabled=not CV2_AVAILABLE,
            help="画像の明るさが不均一な場合に有効です（OpenCV利用時のみ）",
        )

        threshold = st.slider(
            "閾値",
            min_value=0,
            max_value=255,
            value=int(DEFAULT_THRESHOLD),
            step=10,
            disabled=use_adaptive,
            help="2値化の閾値を設定します",
        )

        st.markdown("---")
        st.subheader("OCR精度設定")

        psm_mode = st.selectbox(
            "Page Segmentation Mode",
            options=[
                (0, "Orientation and script detection (OSD) only"),
                (1, "Automatic page segmentation with OSD"),
                (3, "Fully automatic page segmentation, but no OSD"),
                (6, "Assume a single uniform block of text"),
                (11, "Sparse text"),
                (12, "Sparse text with OSD"),
                (13, "Raw line"),
            ],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            index=3,
            help="テキストの配置に応じて最適なモードを選択してください",
        )

        oem_mode = st.selectbox(
            "OCR Engine Mode",
            options=[
                (0, "Legacy engine only"),
                (1, "Neural nets LSTM engine only"),
                (2, "Legacy + LSTM engines"),
                (3, "Default, based on what is available"),
            ],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            index=3,
        )

        st.markdown("---")
        st.subheader("RAG機能")
        if CHROMADB_AVAILABLE:
            st.success("✅ ChromaDB / SentenceTransformers 利用可能")
        else:
            st.warning("⚠️ RAG機能が無効です（依存関係が不足）")
            st.caption(f"import error: {CHROMA_IMPORT_ERROR}")

        st.markdown("---")
        if st.button("🔄 設定をリセット"):
            st.rerun()

    # メインコンテンツ
    tab1, tab2, tab3 = st.tabs(["📤 文書アップロード", "🔍 検索", "📚 文書一覧"])

    # タブ1: 文書アップロード
    with tab1:
        st.header("文書のアップロードとOCR処理")

        uploaded_file = st.file_uploader(
            "PDFまたは画像ファイルをアップロード",
            type=["pdf", "png", "jpg", "jpeg"],
            help="PDFまたは画像ファイルを選択してください",
        )

        if uploaded_file is not None:
            file_details = {
                "ファイル名": uploaded_file.name,
                "ファイルタイプ": uploaded_file.type,
                "ファイルサイズ": f"{uploaded_file.size / 1024:.2f} KB",
            }
            st.json(file_details)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("元の画像")

                file_ext = Path(uploaded_file.name).suffix.lower()

                if file_ext == ".pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    try:
                        # ※ Streamlit Cloudでpoppler不足の場合はここで失敗することがあります
                        images = pdf2image.convert_from_path(tmp_path, dpi=int(PDF_DPI))
                        if images:
                            original_image = images[0]
                            st.image(original_image, caption="PDF 1ページ目", use_container_width=True)
                        else:
                            st.error("PDFから画像を抽出できませんでした")
                            return
                    except Exception as e:
                        st.error(f"PDF処理エラー: {str(e)}")
                        st.info("補足：Streamlit CloudでPDF変換が失敗する場合、PDF→画像変換が必要です（poppler依存）。")
                        return
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                else:
                    original_image = Image.open(uploaded_file)
                    st.image(original_image, caption="アップロード画像", use_container_width=True)

            with col2:
                st.subheader("前処理後の画像")
                processed_image = preprocess_image(
                    original_image,
                    contrast=contrast,
                    brightness=brightness,
                    threshold=threshold,
                    use_adaptive=use_adaptive,
                )
                st.image(processed_image, caption="前処理済み画像", use_container_width=True)

            if st.button("🔍 OCR実行", type="primary"):
                with st.spinner("OCR処理中..."):
                    ocr_result = perform_ocr(
                        processed_image,
                        lang=TESSERACT_LANG,
                        psm=int(psm_mode[0]),
                        oem=int(oem_mode[0]),
                    )

                    st.subheader("OCR結果")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("文字数", ocr_result["char_count"])
                    with c2:
                        st.metric("単語数", ocr_result["word_count"])

                    edited_text = st.text_area(
                        "抽出されたテキスト（編集可）",
                        value=ocr_result["text"],
                        height=300,
                        help="OCRで抽出されたテキストを確認・編集できます",
                    )

                    if st.button("💾 文書を保存", type="primary"):
                        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        doc_info = {
                            "id": doc_id,
                            "filename": uploaded_file.name,
                            "text": edited_text,
                            "char_count": len(edited_text),
                            "word_count": len([w for w in edited_text.split() if w.strip()]),
                            "uploaded_at": datetime.now().isoformat(),
                            "ocr_settings": {
                                "contrast": contrast,
                                "brightness": brightness,
                                "threshold": threshold,
                                "use_adaptive": use_adaptive,
                                "psm": int(psm_mode[0]),
                                "oem": int(oem_mode[0]),
                            },
                        }

                        st.session_state.documents.append(doc_info)

                        # ベクトルDB保存
                        if CHROMADB_AVAILABLE:
                            success = save_document_to_vector_db(
                                doc_id,
                                edited_text,
                                {"filename": uploaded_file.name, "uploaded_at": doc_info["uploaded_at"]},
                            )
                            if success:
                                st.success("✅ 文書がベクトルDBに保存されました")
                            else:
                                st.warning("⚠️ ベクトルDBへの保存に失敗しました")

                        # JSON保存
                        json_path = OCR_RESULTS_DIR / f"{doc_id}.json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(doc_info, f, ensure_ascii=False, indent=2)

                        st.success(f"✅ 文書が保存されました (ID: {doc_id})")
                        st.rerun()

    # タブ2: 検索
    with tab2:
        st.header("RAG検索")

        if not CHROMADB_AVAILABLE:
            st.warning("⚠️ RAG機能を使用するには、ChromaDBとSentenceTransformersが必要です。")
            st.code("pip install chromadb sentence-transformers")
        elif len(st.session_state.documents) == 0:
            st.info("📝 まず文書をアップロードして保存してください。")
        else:
            query = st.text_input(
                "検索クエリを入力",
                placeholder="例: プラスチックの性質について",
                help="検索したい内容を入力してください",
            )

            n_results = st.slider(
                "検索結果数",
                min_value=1,
                max_value=int(MAX_SEARCH_RESULTS),
                value=int(DEFAULT_SEARCH_RESULTS),
            )

            if st.button("🔍 検索実行", type="primary") and query:
                with st.spinner("検索中..."):
                    results = search_vector_db(query, n_results=n_results)

                    if results:
                        st.subheader(f"検索結果 ({len(results)}件)")
                        for i, result in enumerate(results, 1):
                            dist = result.get("distance", None)
                            sim_txt = "N/A" if dist is None else f"{(1.0 - float(dist)):.3f}"
                            with st.expander(f"結果 {i}: {result['id']}（類似度: {sim_txt}）"):
                                st.write("**メタデータ:**")
                                st.json(result.get("metadata", {}))
                                st.write("**テキスト:**")
                                txt = result.get("text", "")
                                st.text(txt[:500] + "..." if len(txt) > 500 else txt)
                    else:
                        st.info("検索結果が見つかりませんでした。")

    # タブ3: 文書一覧
    with tab3:
        st.header("保存済み文書一覧")

        if len(st.session_state.documents) == 0:
            st.info("📝 まだ文書が保存されていません。")
        else:
            st.write(f"**保存済み文書数: {len(st.session_state.documents)}件**")

            for doc in list(st.session_state.documents):
                with st.expander(f"📄 {doc.get('filename','')} ({doc.get('id','')})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**文字数:** {doc.get('char_count',0)}")
                        st.write(f"**単語数:** {doc.get('word_count',0)}")
                    with c2:
                        st.write(f"**アップロード日時:** {doc.get('uploaded_at','')}")

                    st.write("**OCR設定:**")
                    st.json(doc.get("ocr_settings", {}))

                    st.write("**テキスト（一部）:**")
                    text = doc.get("text", "")
                    preview_text = text[:500] + "..." if len(text) > 500 else text
                    st.text(preview_text)

                    if st.button("🗑️ 削除", key=f"delete_{doc.get('id','')}"):
                        st.session_state.documents = [d for d in st.session_state.documents if d.get("id") != doc.get("id")]
                        json_path = OCR_RESULTS_DIR / f"{doc.get('id')}.json"
                        try:
                            if json_path.exists():
                                json_path.unlink()
                        except Exception:
                            pass
                        st.rerun()


if __name__ == "__main__":
    main()
