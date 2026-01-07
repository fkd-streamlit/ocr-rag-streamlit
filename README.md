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
- 🛡️ **PDF容量上限チェック**（メモリ保護機能）
  - PDFファイルサイズ上限チェック（デフォルト: 500MB）
  - PDFページ数上限チェック（デフォルト: 1000ページ）
  - JSONファイルサイズ上限チェック（デフォルト: 100MB）
  - 総チャンク数上限チェック（デフォルト: 50,000チャンク）

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

#### デプロイ手順

1. **GitHubリポジトリの準備**
   - このリポジトリをGitHubにプッシュ
   - `app.py`、`config.py`、`requirements.txt`、`runtime.txt`が含まれていることを確認

2. **Streamlit Cloudでデプロイ**
   - [Streamlit Cloud](https://streamlit.io/cloud)にアクセス
   - GitHubアカウントでログイン
   - 「New app」をクリック
   - リポジトリを選択
   - **Main file path**: `app.py` を指定
   - 「Deploy!」をクリック

3. **JSONファイルの準備**
   - ローカルで `local_ocr_to_json.py` を使ってOCRを実行し、JSONを生成
   - 生成したJSONファイルを `data/ocr_results/` に配置

4. **JSONファイルのアップロード方法（2通り）**
   
   **方法1: GitHubにJSONを配置（小規模向け）**
   - JSONファイルを `data/ocr_results/` に配置してGitHubにプッシュ
   - Streamlit Cloudが自動的に読み込みます
   
   **方法2: アプリ画面からアップロード（推奨）**
   - Streamlitアプリの画面からJSONファイルをアップロード
   - より柔軟で運用が簡単です

> ⚠️ **重要**: Cloud で OCR を実行しないため、TesseractやPopplerは不要です。  
> `requirements.txt` には `streamlit` と `scikit-learn` のみが含まれています。

#### ローカルでテストする場合

```bash
# Streamlit Cloud用の最小限のパッケージでテスト
pip install -r requirements.txt
streamlit run app.py
```

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

ローカルでOCRを実行する場合は、以下のパッケージをインストールしてください：

```bash
# ローカルOCR用の全パッケージ
pip install -r requirements-local.txt

# または個別にインストール
pip install pillow pdf2image pytesseract opencv-python-headless
```

> 💡 **ヒント**: `requirements-local.txt` には、ローカルOCR実行に必要な全パッケージが含まれています。

---

## 📦 ファイル構成

```
ocr-rag-streamlit/
├── app.py                    # Streamlitアプリ（検索・閲覧用）
├── config.py                 # 設定ファイル
├── local_ocr_to_json.py      # ローカルOCRツール
├── requirements.txt          # Streamlit Cloud用（最小限）
├── requirements-local.txt    # ローカルOCR用（全パッケージ）
├── runtime.txt               # Pythonバージョン指定
├── .streamlit/
│   └── config.toml          # Streamlit設定
├── .gitignore               # Git除外設定
└── data/
    └── ocr_results/         # OCR結果JSON保存先
        └── *.json
```

---

## ⚙️ PDF容量上限設定

大きなPDFファイルを処理する際のメモリ保護のため、容量上限チェック機能を実装しています。

### 設定方法

`config.py` で以下の値を調整できます：

```python
# PDF容量上限設定（メモリ保護のため）
MAX_PDF_FILE_SIZE_MB = 500      # PDFファイルサイズ上限（MB）
MAX_PDF_PAGES = 1000            # PDFページ数上限
MAX_JSON_FILE_SIZE_MB = 100    # JSONファイルサイズ上限（MB）
MAX_TOTAL_CHUNKS = 50000       # 総チャンク数上限（メモリ保護）
```

### 動作

- **PDF処理時** (`local_ocr_to_json.py`)
  - 処理前にPDFファイルサイズとページ数をチェック
  - 上限を超える場合はエラーメッセージを表示して処理を中断
  - 進捗表示（10ページごと）で処理状況を確認可能

- **JSON読み込み時** (`app.py`)
  - JSONファイルサイズをチェック
  - 総チャンク数が上限を超える場合は警告を表示
  - 容量超過ファイルは自動的にスキップ

### エラー時の対処法

上限を超える場合は以下のいずれかを試してください：

1. **上限値を調整**: `config.py` の値を大きくする
2. **PDFを分割**: 大きなPDFを複数の小さなPDFに分割して処理
3. **DPIを下げる**: `--dpi` オプションで解像度を下げる（例: `--dpi 150`）
4. **ページ範囲を指定**: `--start` と `--end` で処理範囲を限定

---

## 🚨 トラブルシューティング

### Streamlit Cloudデプロイ時の問題

**問題: デプロイが失敗する**
- `requirements.txt` に `streamlit` と `scikit-learn` が含まれているか確認
- `runtime.txt` にPythonバージョン（例: `3.10`）が指定されているか確認
- `app.py` が正しく配置されているか確認

**問題: JSONファイルが読み込まれない**
- `data/ocr_results/` ディレクトリが存在するか確認
- JSONファイルが正しい形式か確認（`pages` キーが含まれているか）
- ファイルサイズが上限（100MB）を超えていないか確認

**問題: メモリ不足エラー**
- `config.py` の `MAX_TOTAL_CHUNKS` を小さくする
- JSONファイルを分割して処理する
- チャンクサイズを小さくする（サイドバーで調整可能）

### ローカルOCR実行時の問題

**問題: Tesseractが見つからない**
- Tesseractがインストールされているか確認
- 環境変数 `TESSERACT_CMD` を設定
- Windowsの場合、デフォルトパスを確認: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**問題: Popplerが見つからない**
- Popplerがインストールされているか確認
- `bin` ディレクトリがPATHに追加されているか確認

**問題: PDF処理が遅い**
- `--dpi` オプションで解像度を下げる（例: `--dpi 150`）
- `--start` と `--end` で処理範囲を限定
- PDFを分割して処理する

---
