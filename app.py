"""
技術資料OCR・RAG検索アプリケーション
PDF/画像をOCRで読み込み、RAGで検索可能にするWebアプリ
"""

import streamlit as st
import pdf2image
from PIL import Image
import cv2
import numpy as np
import pytesseract
import os
import platform
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple
import tempfile

# Tesseract OCRパス設定
if platform.system() == 'Windows':
    # Windows環境でのTesseract OCRのデフォルトインストールパス
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    
    # 環境変数から取得を試みる
    tesseract_cmd = os.environ.get('TESSERACT_CMD')
    
    if not tesseract_cmd:
        # デフォルトパスを確認
        for path in tesseract_paths:
            if os.path.exists(path):
                tesseract_cmd = path
                break
    
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
else:
    # Linux/Mac環境（Streamlit Cloud含む）
    # packages.txtでインストールされたTesseractは通常 /usr/bin/tesseract に配置される
    # 環境変数から取得を試みる
    tesseract_cmd = os.environ.get('TESSERACT_CMD')
    
    if not tesseract_cmd:
        # Linux環境での一般的なパスを確認
        linux_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',  # macOS (Apple Silicon)
        ]
        
        for path in linux_paths:
            if os.path.exists(path):
                tesseract_cmd = path
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                break
        
        # パスが見つからない場合、whichコマンドで検索を試みる
        if not tesseract_cmd:
            import shutil
            tesseract_path = shutil.which('tesseract')
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        # 環境変数で指定されている場合
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# RAG関連のインポート
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    st.warning("⚠️ ChromaDBがインストールされていません。RAG機能を使用するには `pip install chromadb sentence-transformers` を実行してください。")

# ページ設定
st.set_page_config(
    page_title="技術資料OCR・RAG検索",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# 設定の読み込み
try:
    from config import (
        DATA_DIR, UPLOADS_DIR, OCR_RESULTS_DIR, VECTOR_DB_DIR,
        TESSERACT_LANG, TESSERACT_PSM_DEFAULT, TESSERACT_OEM_DEFAULT,
        PDF_DPI, DEFAULT_CONTRAST, DEFAULT_BRIGHTNESS, DEFAULT_THRESHOLD,
        DEFAULT_USE_ADAPTIVE, VECTOR_DB_COLLECTION_NAME, EMBEDDING_MODEL_NAME,
        DEFAULT_SEARCH_RESULTS, MAX_SEARCH_RESULTS
    )
except ImportError:
    # フォールバック設定
    DATA_DIR = Path("data")
    UPLOADS_DIR = DATA_DIR / "uploads"
    OCR_RESULTS_DIR = DATA_DIR / "ocr_results"
    VECTOR_DB_DIR = DATA_DIR / "chroma_db"
    TESSERACT_LANG = 'jpn'
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
    
    # ディレクトリ作成
    for dir_path in [DATA_DIR, UPLOADS_DIR, OCR_RESULTS_DIR, VECTOR_DB_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def preprocess_image(
    image: Image.Image,
    contrast: float = 1.0,
    brightness: float = 0,
    threshold: int = 127,
    use_adaptive: bool = False
) -> Image.Image:
    """
    画像の前処理（コントラスト、明度、閾値処理）
    
    Args:
        image: PIL Image
        contrast: コントラスト調整値 (0.5-2.0)
        brightness: 明度調整値 (-100 to 100)
        threshold: 閾値 (0-255)
        use_adaptive: 適応的閾値処理を使用するか
    
    Returns:
        前処理済みPIL Image
    """
    # PIL → OpenCV形式に変換
    cv_img = np.array(image.convert('RGB'))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    
    # グレースケール変換
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # コントラスト・明度調整
    if contrast != 1.0 or brightness != 0:
        gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    
    # 閾値処理
    if use_adaptive:
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    else:
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # OpenCV → PIL形式に戻す
    result = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
    return result

def perform_ocr(image: Image.Image, lang: str = "jpn", psm: int = 6, oem: int = 3) -> Dict[str, Any]:
    """
    OCR処理：
    - Tesseractが使えれば pytesseract
    - 使えなければ EasyOCR にフォールバック（Streamlit Cloud向け）
    """
    # まずTesseractを試す
    try:
        custom_config = f"--oem {oem} --psm {psm} -l {lang}"
        text = pytesseract.image_to_string(image, config=custom_config)

        # 成功しているが空文字の場合もあるので軽く判定
        if text and text.strip():
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            return {
                "text": text,
                "data": data,
                "word_count": len([w for w in text.split() if w.strip()]),
                "char_count": len(text),
                "engine": "tesseract",
            }
    except Exception:
        pass

    # Tesseractが無理なら EasyOCR
    try:
        import easyocr
        # reader の生成は重いのでキャッシュ
        @st.cache_resource
        def _get_reader():
            # 日本語+英語（必要なら追加）
            return easyocr.Reader(["ja", "en"], gpu=False)
        reader = _get_reader()

        np_img = np.array(image.convert("RGB"))
        results = reader.readtext(np_img, detail=0)  # text only
        text = "\n".join(results)

        return {
            "text": text,
            "data": {},
            "word_count": len([w for w in text.split() if w.strip()]),
            "char_count": len(text),
            "engine": "easyocr",
        }
    except Exception as e:
        st.error(f"OCRエラー: {str(e)}")
        return {"text": "", "data": {}, "word_count": 0, "char_count": 0, "engine": "none"}


        
        # Tesseract設定
        custom_config = f'--oem {oem} --psm {psm} -l {lang}'
        
        # OCR実行
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # 詳細情報も取得
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        return {
            'text': text,
            'data': data,
            'word_count': len([w for w in text.split() if w.strip()]),
            'char_count': len(text)
        }
    except Exception as e:
        error_msg = str(e)
        # より詳細なエラーメッセージを表示
        if "tesseract is not installed" in error_msg.lower() or "tesseract" in error_msg.lower():
            st.error(
                f"OCRエラー: Tesseract OCRが見つかりません。\n\n"
                f"**Streamlit Cloudの場合:**\n"
                f"- `packages.txt` に以下が含まれているか確認してください:\n"
                f"  - tesseract-ocr\n"
                f"  - tesseract-ocr-jpn\n"
                f"- ファイルがGitHubにプッシュされているか確認してください。\n\n"
                f"**ローカル環境の場合:**\n"
                f"- Tesseract OCRがインストールされているか確認してください。"
            )
        else:
            st.error(f"OCRエラー: {error_msg}")
        return {
            'text': '',
            'data': {},
            'word_count': 0,
            'char_count': 0
        }


def initialize_vector_db():
    """ベクトルDBを初期化"""
    if not CHROMADB_AVAILABLE:
        return None
    
    try:
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(VECTOR_DB_DIR)
        ))
        
        # コレクション取得または作成
        try:
            collection = client.get_collection(VECTOR_DB_COLLECTION_NAME)
        except:
            collection = client.create_collection(VECTOR_DB_COLLECTION_NAME)
        
        return collection
    except Exception as e:
        st.error(f"ベクトルDB初期化エラー: {str(e)}")
        return None


def load_embedding_model():
    """埋め込みモデルを読み込み"""
    if not CHROMADB_AVAILABLE:
        return None
    
        try:
            if st.session_state.embedding_model is None:
                # 日本語対応の埋め込みモデル
                model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                st.session_state.embedding_model = model
            return st.session_state.embedding_model
        except Exception as e:
            st.error(f"埋め込みモデル読み込みエラー: {str(e)}")
            return None


def save_document_to_vector_db(
    doc_id: str,
    text: str,
    metadata: Dict
):
    """文書をベクトルDBに保存"""
    if not CHROMADB_AVAILABLE:
        return False
    
    try:
        collection = initialize_vector_db()
        model = load_embedding_model()
        
        if collection is None or model is None:
            return False
        
        # テキストを埋め込みベクトルに変換
        embeddings = model.encode([text]).tolist()
        
        # ベクトルDBに追加
        collection.add(
            ids=[doc_id],
            embeddings=embeddings,
            documents=[text],
            metadatas=[metadata]
        )
        
        return True
    except Exception as e:
        st.error(f"ベクトルDB保存エラー: {str(e)}")
        return False


def search_vector_db(query: str, n_results: int = 5) -> List[Dict]:
    """ベクトルDBから検索"""
    if not CHROMADB_AVAILABLE:
        return []
    
    try:
        collection = initialize_vector_db()
        model = load_embedding_model()
        
        if collection is None or model is None:
            return []
        
        # クエリを埋め込みベクトルに変換
        query_embedding = model.encode([query]).tolist()
        
        # 検索実行
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # 結果を整形
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                search_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return search_results
    except Exception as e:
        st.error(f"検索エラー: {str(e)}")
        return []


def main():
    """メインアプリケーション"""
    
    # Tesseract OCRのパス確認と表示（Windows環境のみ）
    if platform.system() == 'Windows':
        tesseract_cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            # サイドバーにTesseract情報を表示（後で追加）
            pass
        else:
            # Tesseractが見つからない場合の警告
            st.sidebar.warning(
                f"⚠️ Tesseract OCRが見つかりません。\n\n"
                f"インストールパス: `C:\\Program Files\\Tesseract-OCR`\n\n"
                f"環境変数 `TESSERACT_CMD` を設定するか、\n"
                f"`app.py`のパス設定を確認してください。"
            )
    # Streamlit Cloud（Linux環境）では、packages.txtでTesseractが自動インストールされるため
    # パス設定は不要（システムパスに自動的に追加される）
    
    # タイトル
    st.title("📄 技術資料OCR・RAG検索システム")
    st.markdown("---")
    
    # サイドバー: OCR設定
    with st.sidebar:
        st.header("⚙️ OCR設定")
        
        st.subheader("画像前処理")
        contrast = st.slider(
            "コントラスト",
            min_value=0.5,
            max_value=2.0,
            value=DEFAULT_CONTRAST,
            step=0.1,
            help="画像のコントラストを調整します"
        )
        
        brightness = st.slider(
            "明度",
            min_value=-100,
            max_value=100,
            value=DEFAULT_BRIGHTNESS,
            step=10,
            help="画像の明るさを調整します"
        )
        
        use_adaptive = st.checkbox(
            "適応的閾値処理を使用",
            value=DEFAULT_USE_ADAPTIVE,
            help="画像の明るさが不均一な場合に有効です"
        )
        
        threshold = st.slider(
            "閾値",
            min_value=0,
            max_value=255,
            value=DEFAULT_THRESHOLD,
            step=10,
            disabled=use_adaptive,
            help="2値化の閾値を設定します"
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
                (13, "Raw line")
            ],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            index=3,  # PSM 6に対応
            help="テキストの配置に応じて最適なモードを選択してください"
        )
        
        oem_mode = st.selectbox(
            "OCR Engine Mode",
            options=[
                (0, "Legacy engine only"),
                (1, "Neural nets LSTM engine only"),
                (2, "Legacy + LSTM engines"),
                (3, "Default, based on what is available")
            ],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            index=3  # OEM 3に対応
        )
        
        st.markdown("---")
        st.subheader("システム情報")
        
        # Tesseract OCRのパス表示
        tesseract_cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
        if tesseract_cmd:
            st.success(f"✅ Tesseract OCR: {tesseract_cmd}")
        else:
            st.warning("⚠️ Tesseract OCRのパスが設定されていません")
        
        st.markdown("---")
        st.subheader("その他")
        
        if st.button("🔄 設定をリセット"):
            st.rerun()
    
    # メインコンテンツ
    tab1, tab2, tab3 = st.tabs(["📤 文書アップロード", "🔍 検索", "📚 文書一覧"])
    
    # タブ1: 文書アップロード
    with tab1:
        st.header("文書のアップロードとOCR処理")
        
        uploaded_file = st.file_uploader(
            "PDFまたは画像ファイルをアップロード",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="PDFまたは画像ファイルを選択してください"
        )
        
        if uploaded_file is not None:
            # ファイル情報表示
            file_details = {
                "ファイル名": uploaded_file.name,
                "ファイルタイプ": uploaded_file.type,
                "ファイルサイズ": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("元の画像")
                
                # ファイルを一時保存
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                if file_ext == '.pdf':
                    # PDF処理
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        # PDFを画像に変換
                        images = pdf2image.convert_from_path(tmp_path, dpi=PDF_DPI)
                        if images:
                            original_image = images[0]
                            st.image(original_image, caption="PDF 1ページ目", use_container_width=True)
                        else:
                            st.error("PDFから画像を抽出できませんでした")
                            return
                    except Exception as e:
                        st.error(f"PDF処理エラー: {str(e)}")
                        return
                    finally:
                        os.unlink(tmp_path)
                else:
                    # 画像処理
                    original_image = Image.open(uploaded_file)
                    st.image(original_image, caption="アップロード画像", use_container_width=True)
            
            with col2:
                st.subheader("前処理後の画像")
                
                # 前処理実行
                processed_image = preprocess_image(
                    original_image,
                    contrast=contrast,
                    brightness=brightness,
                    threshold=threshold,
                    use_adaptive=use_adaptive
                )
                st.image(processed_image, caption="前処理済み画像", use_container_width=True)
            
            # OCR実行ボタン
            if st.button("🔍 OCR実行", type="primary"):
                with st.spinner("OCR処理中..."):
                    # OCR実行
                    ocr_result = perform_ocr(
                        processed_image,
                        lang=TESSERACT_LANG,
                        psm=psm_mode[0],
                        oem=oem_mode[0]
                    )
                    
                    # 結果表示
                    st.subheader("OCR結果")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("文字数", ocr_result['char_count'])
                    with col2:
                        st.metric("単語数", ocr_result['word_count'])
                    
                    # OCRテキスト表示
                    st.text_area(
                        "抽出されたテキスト",
                        value=ocr_result['text'],
                        height=300,
                        help="OCRで抽出されたテキストを確認・編集できます"
                    )
                    
                    # 保存ボタン
                    if st.button("💾 文書を保存", type="primary"):
                        # 文書情報を保存
                        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        doc_info = {
                            'id': doc_id,
                            'filename': uploaded_file.name,
                            'text': ocr_result['text'],
                            'char_count': ocr_result['char_count'],
                            'word_count': ocr_result['word_count'],
                            'uploaded_at': datetime.now().isoformat(),
                            'ocr_settings': {
                                'contrast': contrast,
                                'brightness': brightness,
                                'threshold': threshold,
                                'use_adaptive': use_adaptive,
                                'psm': psm_mode[0],
                                'oem': oem_mode[0]
                            }
                        }
                        
                        # セッション状態に追加
                        st.session_state.documents.append(doc_info)
                        
                        # ベクトルDBに保存
                        if CHROMADB_AVAILABLE:
                            success = save_document_to_vector_db(
                                doc_id,
                                ocr_result['text'],
                                {
                                    'filename': uploaded_file.name,
                                    'uploaded_at': doc_info['uploaded_at']
                                }
                            )
                            if success:
                                st.success("✅ 文書がベクトルDBに保存されました")
                            else:
                                st.warning("⚠️ ベクトルDBへの保存に失敗しました")
                        
                        # JSONファイルにも保存
                        json_path = OCR_RESULTS_DIR / f"{doc_id}.json"
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(doc_info, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"✅ 文書が保存されました (ID: {doc_id})")
                        st.rerun()
    
    # タブ2: 検索
    with tab2:
        st.header("RAG検索")
        
        if not CHROMADB_AVAILABLE:
            st.warning("⚠️ RAG機能を使用するには、ChromaDBとSentenceTransformersをインストールしてください。")
            st.code("pip install chromadb sentence-transformers")
        elif len(st.session_state.documents) == 0:
            st.info("📝 まず文書をアップロードして保存してください。")
        else:
            # 検索クエリ入力
            query = st.text_input(
                "検索クエリを入力",
                placeholder="例: プラスチックの性質について",
                help="検索したい内容を入力してください"
            )
            
            n_results = st.slider(
                "検索結果数",
                min_value=1,
                max_value=MAX_SEARCH_RESULTS,
                value=DEFAULT_SEARCH_RESULTS
            )
            
            if st.button("🔍 検索実行", type="primary") and query:
                with st.spinner("検索中..."):
                    results = search_vector_db(query, n_results=n_results)
                    
                    if results:
                        st.subheader(f"検索結果 ({len(results)}件)")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"結果 {i}: {result['id']} (類似度: {1 - result['distance']:.3f} if result['distance'] else 'N/A')"):
                                st.write("**メタデータ:**")
                                st.json(result['metadata'])
                                st.write("**テキスト:**")
                                st.text(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
                    else:
                        st.info("検索結果が見つかりませんでした。")
    
    # タブ3: 文書一覧
    with tab3:
        st.header("保存済み文書一覧")
        
        if len(st.session_state.documents) == 0:
            st.info("📝 まだ文書が保存されていません。")
        else:
            st.write(f"**保存済み文書数: {len(st.session_state.documents)}件**")
            
            for doc in st.session_state.documents:
                with st.expander(f"📄 {doc['filename']} ({doc['id']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**文字数:** {doc['char_count']}")
                        st.write(f"**単語数:** {doc['word_count']}")
                    with col2:
                        st.write(f"**アップロード日時:** {doc['uploaded_at']}")
                    
                    st.write("**OCR設定:**")
                    st.json(doc['ocr_settings'])
                    
                    st.write("**テキスト（一部）:**")
                    preview_text = doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text']
                    st.text(preview_text)
                    
                    if st.button(f"🗑️ 削除", key=f"delete_{doc['id']}"):
                        st.session_state.documents = [d for d in st.session_state.documents if d['id'] != doc['id']]
                        # JSONファイルも削除
                        json_path = OCR_RESULTS_DIR / f"{doc['id']}.json"
                        if json_path.exists():
                            json_path.unlink()
                        st.rerun()


if __name__ == "__main__":
    main()

