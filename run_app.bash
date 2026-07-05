#!/usr/bin/env bash

# 1. 仮想環境（.venv）のパスを指定（スクリプトと同じディレクトリにある想定）
VENV_PATH="./.venv/bin/activate"

# 2. 仮想環境ファイルが存在するかチェック
if [ -f "$VENV_PATH" ]; then
    echo "⚙️ 仮想環境（venv）をアクティベートします..."
    source "$VENV_PATH"
else
    echo "❌ エラー: $VENV_PATH が見つかりません。先に uv venv を実行してください。"
    exit 1
fi

# 3. PythonのGUIアプリを起動
echo "🚀 Rembg GUI アプリケーションを起動中..."
python app.py

# 4. アプリ終了後の処理（deactivate）
if command -v deactivate &> /dev/null; then
    echo "🔌 アプリが終了しました。仮想環境をディアクティベートします。"
    deactivate
else
    echo "👋 アプリが終了しました。"
fi
