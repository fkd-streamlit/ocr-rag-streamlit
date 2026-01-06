# 技術資料OCR・RAG検索システム（ocr-rag-streamlit）

手書き／ワープロ等の技術資料を **OCRでテキスト化** し、**RAG（ベクトル検索）** で検索できる Web アプリです。  
**おすすめ運用（ルートC）** は「OCRはローカルPCで実行し、Streamlit Cloud は“検索・閲覧”に専念」する方式です。

---

## ✅ このリポジトリのおすすめ運用（ルートC）

### なぜルートC？
Streamlit Cloud では **Tesseract のインストールや日本語学習データの依存** が絡みやすく、OCRが不安定になりがちです。  
そのため本プロジェクトでは、以下を推奨します。

- **OCR：ローカルPCで実行（Tesseract / Poppler を自由に使える）**
- **Cloud：JSONを読み込んで検索（RAG）だけを実行**

> つまり、Cloud では「tesseract is not installed」問題を根本回避します。

---

## 機能

- 📄 **PDF/画像アップロード（ローカルOCRツール側）**
- 🎨 **画像前処理**（コントラスト／明度／閾値／適応的閾値など）
- 🧾 **OCR → JSON保存**（ローカルで実行）
- 🔍 **RAG検索**（Streamlit Cloud / ローカルで動作）
- 📚 **文書一覧表示**（保存済みJSONを表示）

---

## 構成（重要）

- `app.py`  
  - **検索・閲覧用のStreamlitアプリ**
  - 推奨：Streamlit Cloud にデプロイするのはこれ

- `local_ocr_to_json.py`（あなたが追加するファイル）  
  - **ローカルPCでOCRを実行して JSON を生成するツール**
  - Streamlit Cloud では実行しません

- `data/ocr_results/*.json`  
  - OCR結果（JSON）
  - Cloud側では「このJSONを読み込んで検索」します

---

## 🚀 まずは動かす（最短）

### A) Streamlit Cloud（検索・閲覧のみ）
1. このリポジトリを Streamlit Cloud にデプロイ
2. JSON（OCR結果）をアプリに読み込ませる  
   - 方式は以下のいずれか  
     - **GitHubにJSONを置く**（小規模向け）  
     - **アプリ画面からJSONをアップロード**（推奨・運用が簡単）

> ※ Cloud で OCR を実行しないため、Tesseractは不要です。

---

### B) ローカルPC（OCR → JSON生成）
ローカルで `local_ocr_to_json.py` を使って OCR し、JSON を生成します。

#### Windowsの前提インストール
1) **Tesseract OCR**
- インストール例：`C:\Program Files\Tesseract-OCR\tesseract.exe`
- 別場所の場合は環境変数 `TESSERACT_CMD` を設定

2) **Poppler（PDF→画像変換用）**
- Poppler をインストールして `bin` を PATH に追加

#### Pythonパッケージ（ローカル用）
```bash
pip install pillow pdf2image pytesseract opencv-python

