# ocr-rag-streamlit
[README.md](https://github.com/user-attachments/files/24452073/README.md)
# 技術資料OCR・RAG検索システム

手書きやワープロの技術資料をOCRで読み込み、RAG（Retrieval-Augmented Generation）で検索可能にするWebアプリケーションです。

## 機能

- 📄 **PDF/画像アップロード**: PDFや画像ファイルをアップロードしてOCR処理
- 🎨 **画像前処理調整**: コントラスト、明度、閾値処理をリアルタイムで調整可能
- 🔍 **OCR精度調整**: Page Segmentation Mode (PSM) や OCR Engine Mode (OEM) を調整
- 💾 **文書保存**: OCR結果をベクトルDBに保存して検索可能に
- 🤖 **RAG検索**: 自然言語クエリで保存された文書を検索
- 👥 **複数人共有**: Streamlit Cloudやローカルサーバーで複数人で利用可能

## セットアップ

### 1. 必要なソフトウェアのインストール

#### Windowsの場合

1. **Tesseract OCR** をインストール
   - [Tesseract OCR公式サイト](https://github.com/UB-Mannheim/tesseract/wiki)からインストーラーをダウンロード
   - デフォルトのインストールパス: `C:\Program Files\Tesseract-OCR`
   - アプリケーションは自動的にこのパスを検出します
   - 別の場所にインストールした場合は、環境変数 `TESSERACT_CMD` に `tesseract.exe` のフルパスを設定してください

2. **Poppler** をインストール（PDF処理用）
   - [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)からダウンロード
   - 解凍後、binフォルダを環境変数PATHに追加

#### Linux/Macの場合

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-jpn poppler-utils

# macOS
brew install tesseract tesseract-lang poppler
```

### 2. Pythonパッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. アプリケーションの起動

#### Windowsの場合
```bash
run.bat
```

#### Linux/Macの場合
```bash
chmod +x run.sh
./run.sh
```

#### 手動起動の場合
```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

## 他の方と共有する方法

アプリを他の方と共有する方法については、以下のガイドを参照してください。

### 🚀 GitHubアカウントをお持ちの場合（推奨）

**最も簡単な方法**: [`GITHUB_DEPLOY.md`](GITHUB_DEPLOY.md) に従って、Streamlit Cloudでデプロイしてください。

**クイック手順**:
1. GitHubでリポジトリを作成
2. `deploy_to_github.bat` を実行してコードをプッシュ
3. [Streamlit Cloud](https://share.streamlit.io/)でデプロイ
4. 生成されたURLを共有

詳細は [`GITHUB_DEPLOY.md`](GITHUB_DEPLOY.md) をご覧ください。

### その他の共有方法

- 🖥️ **ローカルサーバー**（社内ネットワーク向け）: [`SHARING_GUIDE.md`](SHARING_GUIDE.md)
- 🐳 **Docker**（本番環境向け）: [`DEPLOYMENT.md`](DEPLOYMENT.md)

---

## 使い方

### 1. 文書のアップロード

1. **「📤 文書アップロード」**タブを開く
2. PDFまたは画像ファイルをアップロード
3. サイドバーで画像前処理パラメータを調整：
   - **コントラスト**: 画像のコントラストを調整（0.5-2.0）
   - **明度**: 画像の明るさを調整（-100 to 100）
   - **適応的閾値処理**: 明るさが不均一な画像に有効
   - **閾値**: 2値化の閾値を設定（0-255）

### 2. OCR精度の調整

サイドバーの「OCR精度設定」で調整：

- **Page Segmentation Mode (PSM)**: テキストの配置に応じて最適なモードを選択
  - `6`: 単一の均一なテキストブロック（一般的な文書に最適）
  - `11`: まばらなテキスト
  - `13`: 生の行（手書きに適している場合あり）

- **OCR Engine Mode (OEM)**: OCRエンジンのモード
  - `3`: デフォルト（推奨）

### 3. OCR実行と保存

1. 「🔍 OCR実行」ボタンをクリック
2. 抽出されたテキストを確認・編集
3. 「💾 文書を保存」ボタンでベクトルDBに保存

### 4. 検索

1. **「🔍 検索」**タブを開く
2. 検索クエリを入力（例: "プラスチックの性質について"）
3. 「🔍 検索実行」ボタンをクリック
4. 類似度の高い文書が表示されます

## 複数人での共有

### Streamlit Cloudで共有（推奨）

1. GitHubリポジトリにコードをプッシュ
2. [Streamlit Cloud](https://streamlit.io/cloud)にサインアップ
3. リポジトリを接続してデプロイ
4. 共有URLをチームメンバーに共有

### ローカルサーバーで共有

```bash
# サーバーを起動（0.0.0.0で全インターフェースからアクセス可能）
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

ファイアウォール設定でポート8501を開放し、`http://[サーバーIP]:8501` でアクセス可能になります。

## データの保存場所

- **アップロードファイル**: `data/uploads/`
- **OCR結果**: `data/ocr_results/`
- **ベクトルDB**: `data/chroma_db/`

## トラブルシューティング

### Tesseractが見つからないエラー

- Windows: 環境変数PATHにTesseractのインストールパスを追加
- Linux/Mac: `which tesseract` でパスを確認し、必要に応じてPATHを設定

### PDF処理エラー

- Popplerが正しくインストールされているか確認
- 環境変数PATHにPopplerのbinフォルダが含まれているか確認

### メモリ不足エラー

- 大きなPDFファイルは1ページずつ処理することを推奨
- 画像の解像度を下げる（現在は300 DPI）

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 今後の拡張予定

- [ ] 複数ページPDFの一括処理
- [ ] 画像内の図表の検出と抽出
- [ ] より高度な誤字補正機能
- [ ] LLMを使った要約機能
- [ ] エクスポート機能（JSON, CSV, Markdown）

