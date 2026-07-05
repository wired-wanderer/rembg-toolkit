#!/usr/bin/fish

# 1. 仮想環境（.venv）のパスを指定（スクリプトと同じディレクトリにある想定）
set VENV_PATH "./.venv/bin/activate.fish"

# 2. 仮想環境ファイルが存在するかチェック
if test -f $VENV_PATH
    echo "⚙️ 仮想環境（venv）をアクティベートします..."
    source $VENV_PATH
else
    echo "❌ エラー: $VENV_PATH が見つかりません。先に uv venv を実行してください。"
    exit 1
end

# 3. PythonのGUIアプリを起動
echo "🚀 Rembg GUI アプリケーションを起動中..."
python app.py

# 4. アプリ終了後の処理（deactivate）
# ※fishの環境によっては、deactivate関数が定義されているためそれを呼び出します
if functions -q deactivate
    echo "🔌 アプリが終了しました。仮想環境をディアクティベートします。"
    deactivate
else
    echo "👋 アプリが終了しました。"
end
