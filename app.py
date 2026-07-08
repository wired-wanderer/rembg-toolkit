import sys
import os
import io
import math
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QFileDialog, QGroupBox, QFormLayout, 
    QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QProgressBar, QMessageBox, QComboBox, QLineEdit,
    QTextEdit, QTabWidget, QSlider, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor
from rembg import remove, new_session
from rembg.sessions.base import BaseSession
from PIL import Image, ImageFilter
import numpy as np
import shutil
from scipy import ndimage

try:
    from pymatting import estimate_alpha_cf, estimate_foreground_ml
    PYMATTING_AVAILABLE = True
except Exception:
    PYMATTING_AVAILABLE = False


# --- リサンプリング方式の対応表 ---
# 言語に依存しない内部キー(bilinear/bicubic/lanczos3) と PIL.Image.Resampling の
# 定数を紐付ける、アプリ全体で共有する固定マッピング。
# 表示ラベルは TRANSLATIONS["resample_bilinear"] 等、言語ごとに切り替える。
RESAMPLE_KEYS = ["bilinear", "bicubic", "lanczos3"]
RESAMPLE_PIL = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos3": Image.Resampling.LANCZOS,
}

# =========================================================================
# --- 簡易i18n(日本語/English)基盤 ---
# Qt Linguist(.ts/.qm)を使ったフル実装ではなく、辞書ベースの軽量な仕組み。
# 理由: 現状のUIはtr()を一切使わず文字列を直書きしているため、
#       .ts方式に完全移行すると全面書き換えが必要になり工数が跳ね上がる。
#       辞書 + retranslate_ui() で「実行時に言語を切り替えられる」実用最小構成にする。
# =========================================================================
TRANSLATIONS = {
    "window_title": {"ja": "Rembg-toolkit 高機能一括背景透過ツール", "en": "Rembg-toolkit — Advanced Batch Background Remover"},

    "tab_rembg": {"ja": "背景除去", "en": "Background Removal"},
    "tab_upscale": {"ja": "アップスケール", "en": "Upscale"},
    "tab_matting": {"ja": "マットインペイント", "en": "Matte Inpainting"},
    "tab_output": {"ja": "保存先設定", "en": "Output Settings"},
    "placeholder_text": {"ja": "🚧 「{name}」は実装予定です 🚧", "en": "🚧 \"{name}\" is planned for a future release 🚧"},

    "drop_files_label": {"ja": "処理する画像ファイル (ここにドラッグ＆ドロップ)", "en": "Image files to process (drag & drop here)"},
    "btn_add_files": {"ja": "ファイルを追加", "en": "Add Files"},
    "btn_remove_selected": {"ja": "選択削除", "en": "Remove Selected"},
    "btn_clear_list": {"ja": "リストクリア", "en": "Clear List"},
    "preview_group": {"ja": "プレビュー比較 (処理前 / 処理後)", "en": "Preview Comparison (Before / After)"},
    "preview_before_empty": {"ja": "処理前\n(ファイル未選択)", "en": "Before\n(no file selected)"},
    "preview_after_empty": {"ja": "処理後\n(未処理)", "en": "After\n(not processed)"},
    "preview_unavailable": {"ja": "(表示できません)", "en": "(cannot display)"},
    "log_label": {"ja": "処理ログ:", "en": "Processing Log:"},
    "log_placeholder": {"ja": "ステータス: 待機中", "en": "Status: Idle"},

    "preset_group": {"ja": "設定プリセット", "en": "Settings Presets"},
    "preset_name_label": {"ja": "プリセット名:", "en": "Preset Name:"},
    "btn_save": {"ja": "保存", "en": "Save"},
    "btn_load": {"ja": "読み込み", "en": "Load"},
    "btn_refresh": {"ja": "更新", "en": "Refresh"},

    "custom_onnx_group": {"ja": "自前ONNXモデルの使用 (ToonOutなど)", "en": "Use Custom ONNX Model (ToonOut, etc.)"},
    "check_use_custom": {"ja": "自前ONNXモデルを使用する", "en": "Use a custom ONNX model"},
    "model_select_label": {"ja": "モデル選択:", "en": "Select Model:"},
    "btn_browse_onnx": {"ja": "外部からファイルを追加...", "en": "Add File from Disk..."},

    "perf_group": {"ja": "パフォーマンス設定", "en": "Performance Settings"},
    "check_use_batch": {"ja": "バッチ推論を使用する（自前ONNXモデル専用）", "en": "Use batched inference (custom ONNX models only)"},
    "batch_size_label": {"ja": "バッチサイズ:", "en": "Batch Size:"},

    "model_group": {"ja": "標準AIモデルの選択 (自前ONNXが空のとき適用)", "en": "Standard AI Model (used when no custom ONNX is selected)"},
    "model_label": {"ja": "モデル:", "en": "Model:"},

    "resample_group": {"ja": "リサンプリング方式", "en": "Resampling Method"},
    "resample_label": {"ja": "方式:", "en": "Method:"},
    "resample_bilinear": {"ja": "バイリニア", "en": "Bilinear"},
    "resample_bicubic": {"ja": "バイキュービック", "en": "Bicubic"},
    "resample_lanczos3": {"ja": "ランチョス3", "en": "Lanczos-3"},

    "mask_adjust_group": {"ja": "マスクエッジの直接調整", "en": "Direct Mask Edge Adjustment"},
    "mask_blur_label": {"ja": "マスクブラー (Blur):", "en": "Mask Blur:"},
    "mask_offset_label": {"ja": "マスクオフセット (Offset):", "en": "Mask Offset:"},

    "spill_group": {"ja": "背景色スピル除去 (グリーンバック等の色にじみ対策)", "en": "Background Color Spill Suppression (green-screen bleed, etc.)"},
    "spill_enable_label": {"ja": "スピル除去:", "en": "Spill Suppression:"},
    "check_enable": {"ja": "有効にする", "en": "Enable"},
    "spill_threshold_label": {"ja": "色差しきい値:", "en": "Color-Diff Threshold:"},
    "spill_patch_label": {"ja": "背景色サンプル幅(px):", "en": "Sample Patch Size (px):"},

    "hole_fill_group": {"ja": "自動穴埋め処理 (前景内部の孤立した穴を除去)", "en": "Auto Hole-Fill (removes isolated holes in the foreground)"},
    "hole_fill_label": {"ja": "穴埋め処理:", "en": "Hole-Fill:"},
    "hole_fill_threshold_label": {"ja": "前景判定しきい値:", "en": "Foreground Threshold:"},
    "hole_fill_max_area_label": {"ja": "最大穴サイズ(px²):", "en": "Max Hole Size (px²):"},

    "alpha_matting_group": {"ja": "アルファマッティング（AI自動補正）", "en": "Alpha Matting (AI auto-refinement)"},
    "only_mask_label": {"ja": "マスクのみ出力:", "en": "Mask Only Output:"},
    "check_om": {"ja": "有効にする (-om)", "en": "Enable (-om)"},
    "alpha_matting_label": {"ja": "アルファ補正:", "en": "Alpha Refinement:"},
    "bg_threshold_label": {"ja": "背景閾値 (bg):", "en": "Background Threshold (bg):"},
    "fg_threshold_label": {"ja": "前景閾値 (fg):", "en": "Foreground Threshold (fg):"},
    "erode_size_label": {"ja": "侵食サイズ (erode):", "en": "Erode Size:"},

    "btn_run": {"ja": "一括処理（バッチ）を開始", "en": "Start Batch Processing"},
    "btn_cancel": {"ja": "キャンセル", "en": "Cancel"},

    "output_group_rembg": {"ja": "背景除去", "en": "Background Removal"},
    "output_group_upscale": {"ja": "アップスケール", "en": "Upscale"},
    "output_group_matting": {"ja": "マットインペイント", "en": "Matte Inpainting"},
    "output_group_suffix": {"ja": "の出力先", "en": " Output Folder"},
    "output_save_to_label": {"ja": "保存先:", "en": "Save to:"},
    "btn_browse": {"ja": "選択...", "en": "Browse..."},
    "browse_dir_dialog_title": {"ja": "出力先フォルダを選択", "en": "Select Output Folder"},
    "language_group": {"ja": "表示言語 / Language", "en": "Display Language / 表示言語"},
    "language_label": {"ja": "言語:", "en": "Language:"},

    "warning_title": {"ja": "警告", "en": "Warning"},
    "error_title": {"ja": "エラー", "en": "Error"},
    "msg_no_files": {"ja": "処理するファイルがリストに登録されていません。", "en": "No files have been added to the list."},
    "msg_no_output_dir": {"ja": "出力先フォルダが設定されていません。右側の「出力先フォルダ」から設定してください。", "en": "No output folder is set. Please set one from the \"Output Folder\" panel."},
    "msg_output_dir_not_found": {"ja": "指定された出力先フォルダが見つかりません。パスを確認してください。", "en": "The specified output folder was not found. Please check the path."},
    "msg_no_custom_onnx_selected": {"ja": "自前ONNXモデルが選択されていません。onnx/フォルダにファイルを追加してください。", "en": "No custom ONNX model is selected. Please add a file to the onnx/ folder."},
    "msg_no_preset_name": {"ja": "プリセット名を入力してください。", "en": "Please enter a preset name."},
    "msg_preset_not_found": {"ja": "プリセット「{name}」が見つかりません。", "en": "Preset \"{name}\" was not found."},
    "msg_preset_save_failed": {"ja": "プリセットの保存に失敗しました: {err}", "en": "Failed to save preset: {err}"},
    "msg_preset_load_failed": {"ja": "プリセットの読み込みに失敗しました: {err}", "en": "Failed to load preset: {err}"},
    "msg_onnx_copy_failed": {"ja": "ファイルのコピーに失敗しました: {err}", "en": "Failed to copy file: {err}"},
    "log_preset_saved": {"ja": "プリセット「{name}」を保存しました。", "en": "Preset \"{name}\" saved."},
    "log_preset_loaded": {"ja": "プリセット「{name}」を読み込みました。", "en": "Preset \"{name}\" loaded."},
    "log_cancel_requested": {"ja": "⏸ キャンセル要求を送信しました。現在のファイルの処理完了後に停止します。", "en": "⏸ Cancellation requested. Processing will stop after the current file finishes."},
    "log_batch_warning": {"ja": "⚠ バッチ推論を有効にすると、rembgのアルファマッティング後処理は適用されません。", "en": "⚠ When batched inference is enabled, rembg's alpha-matting post-processing will not be applied."},
    "onnx_list_empty": {"ja": "(onnx/ にファイルがありません)", "en": "(no files in onnx/)"},

    # --- アップスケールタブ ---
    "upscale_model_group": {"ja": "モデル選択 (標準/Real-ESRGAN系)", "en": "Model Selection (Standard / Real-ESRGAN family)"},
    "upscale_standard_label": {"ja": "標準モデル:", "en": "Standard Model:"},
    "btn_download_model": {"ja": "モデルをダウンロード", "en": "Download Model"},
    "upscale_model_ready": {"ja": "✅ ダウンロード済み", "en": "✅ Downloaded"},
    "upscale_model_missing": {"ja": "⬇ 未ダウンロード（実行前に取得します）", "en": "⬇ Not downloaded yet (will fetch before running)"},
    "upscale_custom_group": {"ja": "自前アップスケールモデルの使用 (4x-UltraSharp等)", "en": "Use Custom Upscale Model (4x-UltraSharp, etc.)"},
    "upscale_noncommercial_note": {"ja": "⚠ 自前モデルは各自のライセンス条件（商用可否等）を必ずご確認ください。", "en": "⚠ Please verify the license terms (commercial use, etc.) of any custom model yourself."},
    "upscale_tile_group": {"ja": "タイル分割設定 (VRAM対策)", "en": "Tile Settings (VRAM management)"},
    "upscale_tile_size_label": {"ja": "タイルサイズ (px):", "en": "Tile Size (px):"},
    "upscale_overlap_label": {"ja": "オーバーラップ (px):", "en": "Overlap (px):"},
    "upscale_tile_fixed_note": {"ja": "※このモデルは入力サイズ固定のため、タイルサイズは自動調整されます。", "en": "※ This model requires a fixed input size; tile size will be adjusted automatically."},
    "upscale_target_scale_group": {"ja": "出力倍率の微調整 (任意)", "en": "Fine-tune Output Scale (optional)"},
    "upscale_target_scale_label": {"ja": "目標倍率 (0 = モデル本来の倍率):", "en": "Target Scale (0 = model's native scale):"},
    "upscale_run_group": {"ja": "実行", "en": "Run"},
    "msg_model_not_ready": {"ja": "モデルファイルがありません。先に「モデルをダウンロード」を実行してください。", "en": "Model file not found. Please click \"Download Model\" first."},
    "msg_download_confirm": {"ja": "「{name}」をHugging Faceからダウンロードします ({repo})。よろしいですか？", "en": "This will download \"{name}\" from Hugging Face ({repo}). Continue?"},
    "confirm_title": {"ja": "確認", "en": "Confirm"},

    # --- マットインペイントタブ ---
    "matting_source_label": {"ja": "編集対象の出力元:", "en": "Source Output:"},
    "matting_source_rembg": {"ja": "背景除去の出力", "en": "Background Removal Output"},
    "matting_source_upscale": {"ja": "アップスケールの出力", "en": "Upscale Output"},
    "matting_file_list_label": {"ja": "ファイル一覧", "en": "File List"},
    "matting_no_pymatting": {"ja": "⚠ pymattingがインストールされていません。`pip install pymatting`が必要です。", "en": "⚠ pymatting is not installed. Please run `pip install pymatting`."},
    "matting_brush_group": {"ja": "ブラシ設定", "en": "Brush Settings"},
    "matting_brush_size_label": {"ja": "ブラシサイズ:", "en": "Brush Size:"},
    "matting_brush_unknown": {"ja": "不明領域(境界)", "en": "Unknown (boundary)"},
    "matting_brush_fg": {"ja": "確実に前景", "en": "Definite Foreground"},
    "matting_brush_bg": {"ja": "確実に背景", "en": "Definite Background"},
    "btn_undo": {"ja": "元に戻す (Ctrl+Z)", "en": "Undo (Ctrl+Z)"},
    "btn_redo": {"ja": "やり直す (Ctrl+Shift+Z)", "en": "Redo (Ctrl+Shift+Z)"},
    "btn_fit_view": {"ja": "全体表示", "en": "Fit to Window"},
    "check_preview_alpha": {"ja": "アルファ結果をプレビュー", "en": "Preview Alpha Result"},
    "btn_save_matting": {"ja": "保存 (output/matting/)", "en": "Save (output/matting/)"},
    "matting_canvas_empty": {"ja": "左の一覧から画像を選択してください", "en": "Select an image from the list on the left"},
    "log_matting_saved": {"ja": "【成功】マットインペイント結果を保存しました: {name}", "en": "[Success] Matte inpainting result saved: {name}"},
    "log_matting_recompute_failed": {"ja": "【エラー】アルファ再計算に失敗しました: {err}", "en": "[Error] Alpha recomputation failed: {err}"},
    "msg_no_matting_image": {"ja": "編集する画像が選択されていません。", "en": "No image is selected for editing."},
}


def tr(lang: str, key: str, **kwargs) -> str:
    """指定言語のUI文字列を取得する。未登録キーはキー自体を返す(フォールバック)。"""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(lang, entry.get("ja", key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text


# --- 標準AIモデルの説明文(日本語/English) ---
MODEL_DESCRIPTIONS = {
    "u2net": {"ja": "一般的な用途向けの事前学習済みモデル（万能・標準）", "en": "General-purpose pretrained model (all-round / default)"},
    "u2netp": {"ja": "u2netモデルの軽量版。速度重視、省メモリ環境向け", "en": "Lightweight version of u2net. Faster, lower memory footprint"},
    "u2net_human_seg": {"ja": "人間のセグメンテーション用の事前学習済みモデル（髪や体に強い）", "en": "Pretrained model for human segmentation (strong on hair/body)"},
    "u2net_cloth_seg": {"ja": "人間の肖像から衣服（上半身/下半身/全身）の解析・分離を行うモデル", "en": "Segments clothing (upper/lower/full body) from portraits of people"},
    "silueta": {"ja": "u2netと同精度で、サイズを43MBに縮小した軽量化モデル", "en": "Same accuracy as u2net, reduced to a 43MB lightweight model"},
    "isnet-general-use": {"ja": "一般的な用途向けの新しい高精度な事前学習済みモデル", "en": "Newer, high-accuracy general-purpose pretrained model"},
    "isnet-anime": {"ja": "アニメキャラクター・2Dイラストの高精度セグメンテーション特化", "en": "Specialized for high-accuracy segmentation of anime/2D illustrations"},
    "sam": {"ja": "Meta開発。あらゆる用途に対応した汎用セグメンテーションモデル", "en": "Developed by Meta. General-purpose segmentation model for any use case"},
    "birefnet-general": {"ja": "近年追加された非常に強力で高精度な一般用途向けモデル", "en": "Recently added, very powerful, high-accuracy general-purpose model"},
    "birefnet-general-lite": {"ja": "birefnet-generalの軽量化・高速化モデル", "en": "Lightweight, faster version of birefnet-general"},
    "birefnet-portrait": {"ja": "人物の証明写真や顔写真（ポートレート）に特化したモデル", "en": "Specialized for ID photos and portraits of people"},
    "birefnet-dis": {"ja": "高精度な二分画像セグメンテーション（DIS）用モデル", "en": "Model for high-accuracy Dichotomous Image Segmentation (DIS)"},
    "birefnet-hrsod": {"ja": "高解像度な顕著物体検出（HRSOD）用モデル", "en": "Model for High-Resolution Salient Object Detection (HRSOD)"},
    "birefnet-cod": {"ja": "背景に溶け込んだ「隠された物体（COD）」の検出用モデル", "en": "Model for detecting Camouflaged Objects (COD) blended into the background"},
    "birefnet-massive": {"ja": "大規模データセットを用いて訓練された最高峰の大型モデル", "en": "Top-tier large model trained on a massive dataset"},
    "bria-rmbg": {"ja": "BRIA AIによる、最先端の背景除去モデル[非商用]", "en": "State-of-the-art background removal model by BRIA AI [non-commercial]"},
}


# --- BatchProcessThreadのログメッセージ(日本語/English) ---
LOG_T = {
    "loading_custom_model": {"ja": "カスタムONNXモデル（{name}）を読み込み中...", "en": "Loading custom ONNX model ({name})..."},
    "loading_standard_model": {"ja": "標準AIモデル（{name}）を読み込み中...", "en": "Loading standard AI model ({name})..."},
    "model_load_failed": {"ja": "【エラー】モデルの読み込みに失敗しました: {err}", "en": "[Error] Failed to load model: {err}"},
    "batch_mode_notice": {"ja": "バッチ推論モード (バッチサイズ={size}) で処理します。※rembgのアルファマッティング後処理はこのモードではスキップされます。", "en": "Processing in batched inference mode (batch size={size}). Note: rembg's alpha-matting post-processing is skipped in this mode."},
    "cancelled": {"ja": "⏹ キャンセルされました。（{done}/{total} 件処理済み）", "en": "⏹ Cancelled. ({done}/{total} files processed)"},
    "processing": {"ja": "処理中 ({index}/{total}): {name} ...", "en": "Processing ({index}/{total}): {name} ..."},
    "success_saved": {"ja": "【成功】保存先: {name}", "en": "[Success] Saved to: {name}"},
    "error_bomb": {"ja": "【エラー】{name}: 画像サイズが大きすぎます", "en": "[Error] {name}: image size is too large"},
    "error_generic": {"ja": "【エラー】{name}: {err}", "en": "[Error] {name}: {err}"},
    "error_read_failed": {"ja": "【エラー】{name}: 読み込み失敗 ({err})", "en": "[Error] {name}: failed to read ({err})"},
    "batch_inferring": {"ja": "バッチ {num}/{total} を推論中 ({count}枚)...", "en": "Running inference on batch {num}/{total} ({count} images)..."},
    "batch_fallback_notice": {"ja": "【情報】このモデルはバッチ推論(複数枚同時投入)に対応していません（{err}）。以降は1枚ずつ処理します。", "en": "[Info] This model does not support batched inference ({err}). Falling back to one-by-one processing."},
    "batch_inferring_single": {"ja": "バッチ {num}/{total} を1枚ずつ推論中 ({count}枚)...", "en": "Running inference on batch {num}/{total} one image at a time ({count} images)..."},
    "error_mask_failed": {"ja": "【エラー】{name}: マスク生成に失敗しました", "en": "[Error] {name}: mask generation failed"},
    "batch_complete": {"ja": "=== 完了: 処理に成功したファイル数 {count} 個 ===", "en": "=== Done: {count} file(s) processed successfully ==="},

    # --- アップスケール専用ログ ---
    "upscale_loading_model": {"ja": "アップスケールモデル（{name}）を読み込み中...", "en": "Loading upscale model ({name})..."},
    "download_start": {"ja": "モデルをダウンロード中: {name} ...", "en": "Downloading model: {name} ..."},
    "download_done": {"ja": "✅ ダウンロード完了: {name}", "en": "✅ Download complete: {name}"},
    "download_failed": {"ja": "【エラー】ダウンロードに失敗しました: {err}", "en": "[Error] Download failed: {err}"},
    "cuda_fallback_notice": {"ja": "[警告] CUDA EPが有効化されていません。CPUにフォールバックしています。", "en": "[Warning] CUDA EP is not available. Falling back to CPU."},
}


def log_t(lang: str, key: str, **kwargs) -> str:
    entry = LOG_T.get(key)
    if entry is None:
        return key
    text = entry.get(lang, entry.get("ja", key))
    try:
        return text.format(**kwargs)
    except Exception:
        return text

# 自前ONNXファイルを強制的に読み込ませるためのカスタムセッションクラス
# 内部モジュールから直接 BiRefNetSession を強制インポートします

# =========================================================================
# --- 自前ONNXモデル向け: fp16/fp32 自動判定 ＆ 出力レンジ自動正規化 ---
# 学習環境によって入力dtypeや出力側の最終活性化関数(sigmoid/tanh/線形など)が
# 異なるため、モデルファイルから実際の仕様を読み取り、rembg側(CustomOnnxSession)・
# アップスケール側(UpscaleSession)の両方で同じロジックを使い回す。
# =========================================================================
_ORT_DTYPE_MAP = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}


def ort_input_dtype(session, input_index: int = 0) -> np.dtype:
    """ONNXモデルが実際に要求する入力dtypeを取得する(fp16/fp32自動判定)。"""
    type_str = session.get_inputs()[input_index].type
    return _ORT_DTYPE_MAP.get(type_str, np.float32)


def ort_output_dtype(session, output_index: int = 0) -> np.dtype:
    type_str = session.get_outputs()[output_index].type
    return _ORT_DTYPE_MAP.get(type_str, np.float32)


def auto_normalize_output(raw: np.ndarray) -> np.ndarray:
    """モデル出力の値域を実測して[0,1]のfloat32へ自動正規化する(「クリップ」自動判定)。

    自前でpth→onnx変換したモデルは、学習時の最終活性化関数(sigmoid/tanh/線形など)
    によって出力レンジがまちまちで、単純に np.clip(0,1) すると
      - tanh系(-1〜1中心)の場合: 負の値が全て0に潰れて画像が暗くなる
      - 生ピクセル値(0〜255)の場合: ほぼ全て1.0に張り付いて白飛びする
    といった不具合が起きる。min/maxの実測値から代表的な3パターンを判別して補正する。
    """
    raw = raw.astype(np.float32, copy=False)
    lo, hi = float(raw.min()), float(raw.max())
    if lo < -0.15:
        # tanh系: -1〜1付近に分布 → 0〜1へ再マップ
        raw = (raw + 1.0) / 2.0
    elif hi > 1.5:
        # 0〜255等の生ピクセル値域
        raw = raw / 255.0
    return np.clip(raw, 0.0, 1.0)


class CustomOnnxSession(BaseSession):
    """
    自前ONNXファイル(BiRefNet系)を読み込むためのカスタムセッション。

    注意: BaseSession.__init__() をそのまま呼ぶと、内部の download_models() が
    未登録モデル名で呼ばれて失敗する可能性がある。また inner_session を
    「メソッド」のままにすると BaseSession 側の期待する「InferenceSessionインスタンス」
    と衝突するため、ここでは super().__init__() に頼らず自前で完結させる。
    """

    def __init__(self, model_path: str, resample_method=Image.Resampling.LANCZOS):
        self.model_path = model_path
        self.model_name = "custom-birefnet"
        self.resample_method = resample_method

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self._build_providers()

        self.inner_session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
        )

        active_ep = self.inner_session.get_providers()
        print(f"[CustomOnnxSession] Active Execution Provider: {active_ep}")
        if "CUDAExecutionProvider" not in active_ep:
            print("[警告] CUDA EPが有効化されていません。CPUにフォールバックしています。")

        # --- fp16/fp32自動判定 ---
        # 自前でpth→onnx変換したモデルはfp16でエクスポートされている場合があり、
        # float32のまま流し込むと型不一致でエラーになる(または暗黙キャストで精度崩壊する)。
        self.input_dtype = ort_input_dtype(self.inner_session)
        print(f"[CustomOnnxSession] 検出した入力dtype: {self.input_dtype}")

    @staticmethod
    def _build_providers():
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ))
        providers.append("CPUExecutionProvider")
        return providers

    def predict(self, img: Image.Image, *args, **kwargs) -> list[Image.Image]:
        w, h = img.size
        img_resized = img.convert("RGB").resize((1024, 1024), self.resample_method)  # ← 変更
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        img_np = img_np.transpose((2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0).astype(self.input_dtype)  # ← fp16/fp32自動判定

        input_name = self.inner_session.get_inputs()[0].name
        outputs = self.inner_session.run(None, {input_name: img_np})

        pred = outputs[0][0][0]
        # 修正: 出力の値域(sigmoid[0,1]/tanh[-1,1]/線形など)を実測して自動正規化する
        mask_np = (auto_normalize_output(pred) * 255).astype(np.uint8)

        mask_img = Image.fromarray(mask_np, mode="L")
        mask_img = mask_img.resize((w, h), self.resample_method)  # ← 変更
        return [mask_img]
    
    def predict_batch(self, imgs: list) -> list:
        """
        複数枚をまとめてGPUに投げる、自前ONNXモデル専用の最適化。
        モデルが動的バッチ次元をサポートしていない場合は例外が送出されるため、
        呼び出し側でフォールバック処理を用意すること。
        """
        sizes = [img.size for img in imgs]
        batch_np = []
        for img in imgs:
            img_resized = img.convert("RGB").resize((1024, 1024), self.resample_method)  # ← 変更
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            arr = arr.transpose((2, 0, 1))
            batch_np.append(arr)

        batch_tensor = np.stack(batch_np, axis=0).astype(self.input_dtype)  # ← fp16/fp32自動判定 (N, 3, 1024, 1024)

        input_name = self.inner_session.get_inputs()[0].name
        outputs = self.inner_session.run(None, {input_name: batch_tensor})[0]  # (N, 1, 1024, 1024)

        mask_imgs = []
        for i, (w, h) in enumerate(sizes):
            pred = outputs[i][0]
            mask_np = (auto_normalize_output(pred) * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np, mode="L").resize((w, h), self.resample_method)  # ← 変更
            mask_imgs.append(mask_img)

        return mask_imgs

# 背景削除をバックグラウンドで実行するためのスレッドクラス

import traceback  # ファイル先頭に追加

class BatchProcessThread(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, file_paths, output_dir, options, model_name, use_custom_onnx,
                custom_onnx_path, mask_blur, mask_threshold,
                spill_enabled, spill_threshold, spill_patch,
                hole_fill_enabled, hole_fill_threshold, hole_fill_max_area,  # ← 変更
                resample_method, batch_size=1, lang="ja"):
        super().__init__()
        self.lang = lang
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.options = options
        self.model_name = model_name
        self.use_custom_onnx = use_custom_onnx
        self.custom_onnx_path = custom_onnx_path
        self.mask_blur = mask_blur
        self.mask_threshold = mask_threshold
        self.spill_enabled = spill_enabled
        self.spill_threshold = spill_threshold
        self.spill_patch = spill_patch
        self.hole_fill_enabled = hole_fill_enabled
        self.hole_fill_threshold = hole_fill_threshold
        self.hole_fill_max_area = hole_fill_max_area
        self.resample_method = resample_method
        self.batch_size = batch_size
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    @staticmethod
    def _estimate_background_color(img: Image.Image, patch: int = 10) -> np.ndarray:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        patch = max(1, min(patch, h // 2, w // 2))
        corners = [
            arr[0:patch, 0:patch],
            arr[0:patch, w - patch:w],
            arr[h - patch:h, 0:patch],
            arr[h - patch:h, w - patch:w],
        ]
        samples = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
        return np.median(samples, axis=0)

    @staticmethod
    def _suppress_color_spill(orig_img: Image.Image, alpha: Image.Image,
                               threshold: float = 60.0, patch: int = 10) -> Image.Image:
        bg_color = BatchProcessThread._estimate_background_color(orig_img, patch)
        rgb = np.array(orig_img.convert("RGB"), dtype=np.float32)
        diff = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=-1))
        a = np.array(alpha, dtype=np.float32)
        suppression = np.clip(1.0 - (diff / max(threshold, 1e-6)), 0.0, 1.0)
        a = a * (1.0 - suppression)
        return Image.fromarray(a.astype(np.uint8), mode="L")
    
    @staticmethod
    def _fill_alpha_holes(alpha: Image.Image, binarize_threshold: int = 127, max_hole_area: int = 0) -> Image.Image:
        """
        前景に完全に囲まれ、外部の背景領域と繋がっていない「孤立した穴」だけを不透明化する。
        スピル除去によって作られた境界付近の低アルファ領域(外部と繋がっている)には
        一切干渉しないため、スピル除去との相性問題が発生しない。

        max_hole_area: 0の場合は穴の大きさを問わず全て埋める。
                    正の値を指定すると、その面積(px)を超える穴は
                    意図的な透過部分(服の隙間等)とみなして埋めない。
        """
        arr = np.array(alpha, dtype=np.uint8)
        binary_fg = arr >= binarize_threshold

        filled = ndimage.binary_fill_holes(binary_fg)
        holes_mask = filled & (~binary_fg)  # 埋められた=元々背景判定だった孤立穴

        if max_hole_area > 0:
            labeled_holes, num_holes = ndimage.label(holes_mask)
            for hole_id in range(1, num_holes + 1):
                area = np.sum(labeled_holes == hole_id)
                if area > max_hole_area:
                    holes_mask[labeled_holes == hole_id] = False  # 大きすぎる穴は埋めない

        result = arr.copy()
        result[holes_mask] = 255
        return Image.fromarray(result, mode="L")

    # --- 共通後処理ヘルパー ---
    def _postprocess_alpha(self, orig_img: Image.Image, alpha: Image.Image) -> Image.Image:
        if self.spill_enabled:
            alpha = self._suppress_color_spill(
                orig_img, alpha, threshold=self.spill_threshold, patch=self.spill_patch
            )
        # --- 追加: 穴埋め処理はスピル除去の後、オフセット/ブラーの前に実行 ---
        if self.hole_fill_enabled:
            alpha = self._fill_alpha_holes(
                alpha, binarize_threshold=self.hole_fill_threshold, max_hole_area=self.hole_fill_max_area
            )
        if self.mask_threshold != 0:
            arr = np.array(alpha, dtype=np.int16)
            arr = np.clip(arr + self.mask_threshold, 0, 255).astype(np.uint8)
            alpha = Image.fromarray(arr, mode="L")
        if self.mask_blur > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=self.mask_blur))
        return alpha

    def _save_result(self, file_path: str, orig_img: Image.Image, alpha: Image.Image) -> str:
        pure_name = Path(file_path).stem
        if self.options.get("only_mask"):
            final_img = Image.merge("RGB", (alpha, alpha, alpha))
            out_path = os.path.join(self.output_dir, f"{pure_name}_mask.png")
        else:
            if orig_img.size != alpha.size:
                orig_img = orig_img.resize(alpha.size, Image.Resampling.LANCZOS)
            orig_img.putalpha(alpha)
            final_img = orig_img
            out_path = os.path.join(self.output_dir, f"{pure_name}_rembg.png")
        final_img.save(out_path, "PNG")
        return out_path

    # --- 従来の1枚ずつ処理(標準モデル向け・alpha_matting有効) ---
    def _run_sequential(self, session, total) -> int:
        success_count = 0
        for index, file_path in enumerate(self.file_paths):
            if self._is_cancelled:
                self.log_signal.emit(log_t(self.lang, "cancelled", done=index, total=total))
                break
            try:
                self.log_signal.emit(log_t(self.lang, "processing", index=index + 1, total=total, name=os.path.basename(file_path)))
                with open(file_path, "rb") as f:
                    input_bytes = f.read()

                rembg_options = self.options.copy()
                rembg_options["only_mask"] = False
                output_bytes = remove(input_bytes, session=session, **rembg_options)

                img = Image.open(io.BytesIO(output_bytes))
                alpha = img.getchannel('A')
                orig_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")

                alpha = self._postprocess_alpha(orig_img, alpha)
                out_path = self._save_result(file_path, orig_img, alpha)

                success_count += 1
                self.log_signal.emit(log_t(self.lang, "success_saved", name=os.path.basename(out_path)))
            except Image.DecompressionBombError:
                msg = log_t(self.lang, "error_bomb", name=os.path.basename(file_path))
                self.log_signal.emit(msg); print(msg)
            except Exception as e:
                msg = log_t(self.lang, "error_generic", name=os.path.basename(file_path), err=str(e))
                self.log_signal.emit(msg); print(msg); print(traceback.format_exc())

            self.progress_signal.emit(int(((index + 1) / total) * 100))
        return success_count

    # --- バッチ推論処理(自前ONNXモデル専用・alpha_mattingはバイパス) ---
    def _run_batched(self, session, total) -> int:
        success_count = 0
        processed = 0
        total_batches = (total + self.batch_size - 1) // self.batch_size
        batch_supported = True  # ← 追加: このセッションでバッチ推論が有効かどうか

        for batch_start in range(0, total, self.batch_size):
            if self._is_cancelled:
                self.log_signal.emit(log_t(self.lang, "cancelled", done=processed, total=total))
                break

            batch_paths = self.file_paths[batch_start: batch_start + self.batch_size]
            batch_num = batch_start // self.batch_size + 1

            orig_imgs, valid_paths = [], []
            for file_path in batch_paths:
                try:
                    with open(file_path, "rb") as f:
                        input_bytes = f.read()
                    orig_imgs.append(Image.open(io.BytesIO(input_bytes)).convert("RGB"))
                    valid_paths.append(file_path)
                except Exception as e:
                    processed += 1
                    msg = log_t(self.lang, "error_read_failed", name=os.path.basename(file_path), err=str(e))
                    self.log_signal.emit(msg); print(msg)
                    self.progress_signal.emit(int((processed / total) * 100))

            if not orig_imgs:
                continue

            mask_imgs = None
            if batch_supported:
                self.log_signal.emit(log_t(self.lang, "batch_inferring", num=batch_num, total=total_batches, count=len(orig_imgs)))
                try:
                    mask_imgs = session.predict_batch(orig_imgs)
                except Exception as e:
                    # --- 変更: このモデルはバッチ非対応と判断し、以降は最初から個別推論に切り替える ---
                    batch_supported = False
                    msg = log_t(self.lang, "batch_fallback_notice", err=str(e).splitlines()[0])
                    self.log_signal.emit(msg)
                    print(msg)

            if mask_imgs is None:
                self.log_signal.emit(log_t(self.lang, "batch_inferring_single", num=batch_num, total=total_batches, count=len(orig_imgs)))
                mask_imgs = []
                for img in orig_imgs:
                    try:
                        mask_imgs.append(session.predict(img)[0])
                    except Exception as e2:
                        mask_imgs.append(None)
                        print(f"個別推論も失敗: {e2}")

            for file_path, orig_img, mask_img in zip(valid_paths, orig_imgs, mask_imgs):
                processed += 1
                if mask_img is None:
                    self.log_signal.emit(log_t(self.lang, "error_mask_failed", name=os.path.basename(file_path)))
                    self.progress_signal.emit(int((processed / total) * 100))
                    continue
                try:
                    alpha = self._postprocess_alpha(orig_img, mask_img)
                    out_path = self._save_result(file_path, orig_img, alpha)
                    success_count += 1
                    self.log_signal.emit(log_t(self.lang, "success_saved", name=os.path.basename(out_path)))
                except Exception as e:
                    msg = log_t(self.lang, "error_generic", name=os.path.basename(file_path), err=str(e))
                    self.log_signal.emit(msg); print(msg); print(traceback.format_exc())

                self.progress_signal.emit(int((processed / total) * 100))

        return success_count

    def run(self):
        total = len(self.file_paths)
        try:
            if self.use_custom_onnx and self.custom_onnx_path and os.path.exists(self.custom_onnx_path):
                self.log_signal.emit(log_t(self.lang, "loading_custom_model", name=os.path.basename(self.custom_onnx_path)))
                session = CustomOnnxSession(self.custom_onnx_path, resample_method=self.resample_method)
            else:
                self.log_signal.emit(log_t(self.lang, "loading_standard_model", name=self.model_name))
                session = new_session(self.model_name)
        except Exception as e:
            msg = log_t(self.lang, "model_load_failed", err=str(e))
            self.log_signal.emit(msg); print(msg); print(traceback.format_exc())
            self.finished_signal.emit(0)
            return

        use_batch = self.use_custom_onnx and isinstance(session, CustomOnnxSession) and self.batch_size > 1

        if use_batch:
            self.log_signal.emit(log_t(self.lang, "batch_mode_notice", size=self.batch_size))
            success_count = self._run_batched(session, total)
        else:
            success_count = self._run_sequential(session, total)

        del session
        import gc
        gc.collect()

        self.finished_signal.emit(success_count)


# =========================================================================
# --- アップスケーラー機能 (Real-ESRGAN系, タイル分割 + IOBinding最適化) ---
# =========================================================================

# 標準搭載モデル(BSD-3-Clause系/商用利用可、初回選択時にHugging Faceから自動ダウンロード)
# 注意: qualcomm/Real-ESRGAN-x4plus はNPU向けエクスポートのため入力サイズが
#       固定されている場合がある。UpscaleSession側で自動検出しタイルサイズを追従させる。
UPSCALE_STANDARD_MODELS = {
    "RealESRGAN_x4plus": {
        "repo_id": "qualcomm/Real-ESRGAN-x4plus",
        "filename": "Real-ESRGAN-x4plus.onnx",
        "scale": 4,
        "commercial": True,
    },
    "RealESRGAN_x2plus": {
        "repo_id": "tidus2102/Real-ESRGAN",
        "filename": "Real-ESRGAN_x2plus.onnx",
        "scale": 2,
        "commercial": True,
    },
}


class ModelDownloadThread(QThread):
    """標準アップスケールモデルをHugging Face Hubから非同期ダウンロードする。

    hf_hub_download()内部のtqdm出力に頼ると正確な%進捗をUIに渡せないため、
    requestsで直接ストリーミングDLし、チャンクごとにprogress_signalを発行する。
    """
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # (成功したか, 保存先パス or エラーメッセージ)

    def __init__(self, repo_id: str, filename: str, dest_dir: str, lang: str = "ja"):
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        self.dest_dir = dest_dir
        self.lang = lang

    def run(self):
        try:
            from huggingface_hub import hf_hub_url
            import requests

            url = hf_hub_url(repo_id=self.repo_id, filename=self.filename)
            dest_path = os.path.join(self.dest_dir, self.filename)
            tmp_path = dest_path + ".part"

            self.log_signal.emit(log_t(self.lang, "download_start", name=self.filename))

            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            self.progress_signal.emit(int(downloaded / total * 100))

            os.replace(tmp_path, dest_path)
            self.log_signal.emit(log_t(self.lang, "download_done", name=self.filename))
            self.finished_signal.emit(True, dest_path)
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class UpscaleSession:
    """Real-ESRGAN系ONNXモデル用の、タイル分割 + IOBinding最適化推論セッション。

    GPU最適化の要点:
    - 端の欠けタイルもreflectパディングで「常にtile_size四方」に揃え、全タイルの
      形状を完全固定化する。これにより入出力用GPUバッファ(OrtValue)を最初の1回だけ
      確保し、以降は update_inplace() でデータ差し替えのみを行う
      (タイルごとのcudaMalloc/Free、ホスト⇔デバイスの往復コピーを最小化)。
    - CUDA Execution Providerを優先し、利用不可時はCPUへ自動フォールバックする。
    - タイル境界はコサイン窓による加重ブレンドで重ね合わせ、段差を目立たなくする。
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        use_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers
            else ["CPUExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in providers:
            print(log_t("ja", "cuda_fallback_notice"))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=use_providers)
        self.provider = self.session.get_providers()[0]

        in0 = self.session.get_inputs()[0]
        self.input_name = in0.name
        self.output_name = self.session.get_outputs()[0].name

        # 入力形状が固定(NCHWの全次元が整数)かどうかを検出する。
        # 一部のNPU向けエクスポートは動的軸を持たず固定サイズしか受け付けない。
        shape = in0.shape  # 例: [1, 3, 'height', 'width'] または [1, 3, 128, 128]
        self.fixed_hw = None
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.fixed_hw = (shape[2], shape[3])

        self._io_binding = None
        self._bound_tile_hw = None  # IOBinding用バッファを確保済みのタイルサイズ
        self._in_ort = None
        self._out_ort = None
        self.scale = None  # 初回タイル推論時に出力/入力の空間比から実測する

        # --- fp16/fp32自動判定 ---
        # 自前でpth→onnx変換したモデルはfp16でエクスポートされている場合があり、
        # float32のまま流し込むと型不一致エラーや精度崩壊(暗転など)の原因になる。
        self.input_dtype = ort_input_dtype(self.session)
        self.output_dtype = ort_output_dtype(self.session)
        print(f"[UpscaleSession] 検出した入出力dtype: in={self.input_dtype}, out={self.output_dtype}")

    def _ensure_io_binding(self, tile_h: int, tile_w: int):
        """タイルサイズが前回と同じならGPU入出力バッファを使い回す。"""
        if self._bound_tile_hw == (tile_h, tile_w) and self._io_binding is not None:
            return
        import onnxruntime as ort

        device = "cuda" if self.provider == "CUDAExecutionProvider" else "cpu"
        dummy_in = np.zeros((1, 3, tile_h, tile_w), dtype=self.input_dtype)
        out_h, out_w = tile_h * self.scale, tile_w * self.scale
        dummy_out = np.zeros((1, 3, out_h, out_w), dtype=self.output_dtype)

        self._in_ort = ort.OrtValue.ortvalue_from_numpy(dummy_in, device, 0)
        self._out_ort = ort.OrtValue.ortvalue_from_numpy(dummy_out, device, 0)

        self._io_binding = self.session.io_binding()
        self._io_binding.bind_ortvalue_input(self.input_name, self._in_ort)
        self._io_binding.bind_ortvalue_output(self.output_name, self._out_ort)
        self._bound_tile_hw = (tile_h, tile_w)

    def _infer_tile(self, tile_np: np.ndarray) -> np.ndarray:
        """tile_np: (H, W, 3) float32 [0,1] -> (H*scale, W*scale, 3) float32 [0,1]"""
        h, w = tile_np.shape[:2]
        chw = np.ascontiguousarray(np.transpose(tile_np, (2, 0, 1))[None, ...].astype(self.input_dtype))

        if self.scale is None:
            # 初回のみ通常run()で出力形状(=倍率)を実測してから、以降はIOBinding経路に切り替える
            out = self.session.run([self.output_name], {self.input_name: chw})[0]
            self.scale = out.shape[2] // h
        else:
            self._ensure_io_binding(h, w)
            self._in_ort.update_inplace(chw)
            self.session.run_with_iobinding(self._io_binding)
            out = self._out_ort.numpy()

        # 出力の値域(sigmoid[0,1]/tanh[-1,1]/線形0-255など)を実測して自動正規化する。
        # ここを固定 np.clip(0,1) にしていると、tanh系モデルで負値が0に潰れて暗転する。
        out = auto_normalize_output(out[0])
        return np.transpose(out, (1, 2, 0))  # (H*scale, W*scale, 3)

    @staticmethod
    def _blend_weight(size: int, overlap: int) -> np.ndarray:
        """タイル境界のつなぎ目を目立たなくする、コサイン窓による2Dブレンド重み。"""
        ramp = np.ones(size, dtype=np.float32)
        if overlap > 0:
            t = np.linspace(0, np.pi / 2, overlap, dtype=np.float32)
            edge = np.sin(t) ** 2
            ramp[:overlap] = edge
            ramp[-overlap:] = edge[::-1]
        return ramp[:, None] * ramp[None, :]

    def upscale(self, img: Image.Image, tile_size: int = 512, overlap: int = 32,
                progress_cb=None, cancel_cb=None) -> Image.Image:
        # 固定入力サイズのモデルは、そのサイズに強制的に合わせる
        if self.fixed_hw is not None:
            tile_size = self.fixed_hw[0]
            overlap = min(overlap, tile_size // 4)

        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        stride = max(1, tile_size - overlap)
        n_tiles_y = max(1, math.ceil(max(0, h - overlap) / stride))
        n_tiles_x = max(1, math.ceil(max(0, w - overlap) / stride))
        pad_h = max(0, n_tiles_y * stride + overlap - h)
        pad_w = max(0, n_tiles_x * stride + overlap - w)

        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        total_tiles = n_tiles_y * n_tiles_x
        done = 0
        out_canvas = weight_canvas = weight_tile = None

        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                if cancel_cb and cancel_cb():
                    raise RuntimeError("cancelled")
                y0, x0 = ty * stride, tx * stride
                tile = padded[y0:y0 + tile_size, x0:x0 + tile_size]

                up_tile = self._infer_tile(tile)
                scale = self.scale

                if out_canvas is None:
                    out_canvas = np.zeros((padded.shape[0] * scale, padded.shape[1] * scale, 3), dtype=np.float32)
                    weight_canvas = np.zeros((padded.shape[0] * scale, padded.shape[1] * scale, 1), dtype=np.float32)
                    weight_tile = self._blend_weight(tile_size * scale, overlap * scale)[..., None]

                oy0, ox0 = y0 * scale, x0 * scale
                out_canvas[oy0:oy0 + tile_size * scale, ox0:ox0 + tile_size * scale] += up_tile * weight_tile
                weight_canvas[oy0:oy0 + tile_size * scale, ox0:ox0 + tile_size * scale] += weight_tile

                done += 1
                if progress_cb:
                    progress_cb(int(done / total_tiles * 100))

        weight_canvas[weight_canvas == 0] = 1.0
        result = out_canvas / weight_canvas
        result = result[: h * self.scale, : w * self.scale]
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def close(self):
        self._io_binding = None
        self._in_ort = None
        self._out_ort = None
        del self.session
        import gc
        gc.collect()


class UpscaleProcessThread(QThread):
    """既存のBatchProcessThreadと同じUXパターン(進捗/ログ/キャンセル/完了シグナル)を踏襲。"""
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, file_paths, output_dir, model_path, tile_size, overlap,
                 target_scale=None, resample_method=Image.Resampling.LANCZOS, lang="ja"):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.model_path = model_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.target_scale = target_scale  # Noneまたは0なら「モデルのネイティブ倍率のまま」
        self.resample_method = resample_method
        self.lang = lang
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        total = len(self.file_paths)
        success_count = 0
        try:
            self.log_signal.emit(log_t(self.lang, "upscale_loading_model", name=os.path.basename(self.model_path)))
            session = UpscaleSession(self.model_path)
        except Exception as e:
            msg = log_t(self.lang, "model_load_failed", err=str(e))
            self.log_signal.emit(msg); print(msg); print(traceback.format_exc())
            self.finished_signal.emit(0)
            return

        for index, file_path in enumerate(self.file_paths):
            if self._is_cancelled:
                self.log_signal.emit(log_t(self.lang, "cancelled", done=index, total=total))
                break
            try:
                self.log_signal.emit(log_t(self.lang, "processing", index=index + 1, total=total, name=os.path.basename(file_path)))
                img = Image.open(file_path).convert("RGB")

                def _progress(p, i=index, t=total):
                    self.progress_signal.emit(int(((i + p / 100.0) / t) * 100))

                result = session.upscale(
                    img, tile_size=self.tile_size, overlap=self.overlap,
                    progress_cb=_progress, cancel_cb=lambda: self._is_cancelled,
                )

                if self.target_scale and session.scale and abs(self.target_scale - session.scale) > 1e-3:
                    new_w, new_h = int(img.width * self.target_scale), int(img.height * self.target_scale)
                    result = result.resize((new_w, new_h), self.resample_method)

                stem = Path(file_path).stem
                out_path = os.path.join(self.output_dir, f"{stem}_upscaled.png")
                result.save(out_path)
                success_count += 1
                self.log_signal.emit(log_t(self.lang, "success_saved", name=os.path.basename(out_path)))
            except Exception as e:
                if str(e) == "cancelled":
                    self.log_signal.emit(log_t(self.lang, "cancelled", done=index, total=total))
                    break
                msg = log_t(self.lang, "error_generic", name=os.path.basename(file_path), err=str(e))
                self.log_signal.emit(msg); print(msg); print(traceback.format_exc())

            self.progress_signal.emit(int(((index + 1) / total) * 100))

        session.close()
        self.finished_signal.emit(success_count)


# ドラッグ＆ドロップ対応のリストウィジェット
class DropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    items = [self.item(i).text() for i in range(self.count())]
                    if file_path not in items:
                        self.addItem(file_path)


# =========================================================================
# --- マットインペイント: Trimapベースのアルファマッティング編集キャンバス ---
# =========================================================================
class MattingCanvas(QWidget):
    """QPainterベースのTrimap編集キャンバス。

    操作方法:
      - 左ドラッグ: 現在のブラシモード(不明/前景/背景)でTrimapを塗る
      - マウスホイール: カーソル位置基準でズーム
      - 右ドラッグ / 中ボタンドラッグ: パン
      - Ctrl+Z / Ctrl+Shift+Z: Undo / Redo

    パフォーマンス方針:
      ストローク終了時、なぞった範囲のバウンディングボックス(+マージン)だけを
      切り出してpymattingで再計算する。画像全体を毎回再計算しないため、
      高解像度画像でも編集操作がサクサク動く。
    """
    status_changed = pyqtSignal()  # Undo/Redo可否、保存可否などの外部UI更新用
    brush_size_changed = pyqtSignal(int)

    BRUSH_BG = 0
    BRUSH_UNKNOWN = 128
    BRUSH_FG = 255

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(300, 300)

        self.image_arr = None        # (H,W,3) uint8 元画像(RGB)
        self.trimap = None           # (H,W) uint8   0=BG / 128=不明 / 255=FG
        self.alpha = None            # (H,W) float32 [0,1] 現在のアルファ
        self.foreground_arr = None   # (H,W,3) uint8 デコンタミ後の前景色

        self.zoom = 1.0
        self.origin = QPointF(0, 0)  # 画像(0,0)がウィジェット座標のどこに来るか
        self._user_adjusted_view = False  # ユーザーが手動でズーム/パンしたらリサイズ時の自動fitを止める
        self._panning = False
        self._pan_anchor = None
        self._drawing = False
        self._last_img_pt = None
        self._stroke_bbox = None     # (x0,y0,x1,y1) 現在のストロークで塗った範囲
        self._last_widget_pt = None

        self.brush_size = 30         # 画像ピクセル単位の半径
        self.brush_mode = self.BRUSH_UNKNOWN
        self.preview_alpha = False

        self._base_qimage = None
        self._overlay_dirty = True
        self._overlay_qimage = None
        self._preview_dirty = True
        self._preview_qimage = None

        self._undo_stack = []
        self._redo_stack = []
        self._max_undo = 30

        self.recompute_error = None  # UI側でログ出力するためのエラー保持

    # ------------------------------------------------------------------
    # --- 画像読み込み / 初期Trimap生成 ---
    # ------------------------------------------------------------------
    def load_image(self, pil_img: Image.Image):
        rgba = pil_img.convert("RGBA")
        arr = np.asarray(rgba)
        self.image_arr = np.ascontiguousarray(arr[..., :3])
        if arr[..., 3].min() < 255:
            self.alpha = arr[..., 3].astype(np.float32) / 255.0
        else:
            self.alpha = np.ones(arr.shape[:2], dtype=np.float32)
        self.foreground_arr = self.image_arr.copy()
        self.trimap = self._trimap_from_alpha(self.alpha)

        self._undo_stack.clear()
        self._redo_stack.clear()
        self._base_qimage = None
        self._overlay_dirty = True
        self._preview_dirty = True
        self.fit_to_window()
        self.update()
        self.status_changed.emit()

    @staticmethod
    def _trimap_from_alpha(alpha: np.ndarray) -> np.ndarray:
        """既存のアルファチャンネルから初期Trimapを自動生成する。
        alpha≈0/1の内部は確定領域として扱い、境界付近だけを「不明」帯として残す。
        アルファが全く無い(全て1.0)画像は全域「不明」として、ユーザーの手塗りに委ねる。
        """
        trimap = np.full(alpha.shape, 128, dtype=np.uint8)
        trimap[alpha > 0.95] = 255
        trimap[alpha < 0.05] = 0
        return trimap

    def has_image(self) -> bool:
        return self.image_arr is not None

    # ------------------------------------------------------------------
    # --- 表示制御 ---
    # ------------------------------------------------------------------
    def fit_to_window(self):
        if self.image_arr is None:
            return
        h, w = self.image_arr.shape[:2]
        margin = 20
        zx = max(self.width() - margin, 50) / w
        zy = max(self.height() - margin, 50) / h
        self.zoom = max(0.02, min(zx, zy, 8.0))
        self.origin = QPointF(
            (self.width() - w * self.zoom) / 2,
            (self.height() - h * self.zoom) / 2,
        )
        self._user_adjusted_view = False
        self.update()

    def set_preview_alpha(self, enabled: bool):
        self.preview_alpha = enabled
        self._preview_dirty = True
        self.update()

    def set_brush_mode(self, mode: int):
        self.brush_mode = mode

    def set_brush_size(self, size: int):
        if self.brush_size != size:
            self.brush_size = size
            self.update()
            self.brush_size_changed.emit(size)

    # ------------------------------------------------------------------
    # --- Undo / Redo (Trimap全体のスナップショット、最低限のスタック実装) ---
    # ------------------------------------------------------------------
    def _push_undo(self):
        if self.trimap is None:
            return
        self._undo_stack.append(self.trimap.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self.status_changed.emit()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self.trimap.copy())
        self.trimap = self._undo_stack.pop()
        self._recompute_full()
        self.status_changed.emit()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self.trimap.copy())
        self.trimap = self._redo_stack.pop()
        self._recompute_full()
        self.status_changed.emit()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    # ------------------------------------------------------------------
    # --- pymattingによるアルファ再計算 (バウンディングボックス限定) ---
    # ------------------------------------------------------------------
# 変更後 (最適化および堅牢化リファクタリング)
    def _recompute_region(self, bbox):
        if self.image_arr is None:
            return

        x0, y0, x1, y1 = bbox
        margin = max(8, self.brush_size)
        h, w = self.image_arr.shape[:2]
        x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
        x1 = min(w, x1 + margin); y1 = min(h, y1 + margin)
        if x1 <= x0 or y1 <= y0:
            return

        trimap_crop = self.trimap[y0:y1, x0:x1]

        if not PYMATTING_AVAILABLE:
            self.alpha[y0:y1, x0:x1] = (trimap_crop.astype(np.float32) / 255.0)
            self.recompute_error = "pymatting_missing"
            self._overlay_dirty = True
            self._preview_dirty = True
            return

        # pymattingの例外防止のための事前判定: トリマップ内に既知の背景(0)と前景(255)が
        # 両方存在しない場合、estimate_alpha_cfは解けずValueErrorになる。
        # 不明領域そのものが無い場合も呼ぶ意味が無い。どちらのケースも
        # トリマップ通りの硬いカットオフだけを即座に反映し、応答性を優先する。
        has_unknown = np.any(trimap_crop == self.BRUSH_UNKNOWN)
        has_bg = np.any(trimap_crop == self.BRUSH_BG)
        has_fg = np.any(trimap_crop == self.BRUSH_FG)

        if not has_unknown or not (has_bg and has_fg):
            self.alpha[y0:y1, x0:x1] = (trimap_crop.astype(np.float32) / 255.0)
            self.recompute_error = None
            self._overlay_dirty = True
            self._preview_dirty = True
            return

        try:
            img_crop = self.image_arr[y0:y1, x0:x1].astype(np.float64) / 255.0
            tri_crop = trimap_crop.astype(np.float64) / 255.0
            alpha_crop = estimate_alpha_cf(img_crop, tri_crop)
            fg_crop, _bg_crop = estimate_foreground_ml(img_crop, alpha_crop, return_background=True)
            self.alpha[y0:y1, x0:x1] = alpha_crop.astype(np.float32)
            self.foreground_arr[y0:y1, x0:x1] = np.clip(fg_crop * 255.0, 0, 255).astype(np.uint8)
            self.recompute_error = None
        except Exception as e:
            # 例外時に何も更新しないと「保存しても無反応」に見えるため、
            # 最低限トリマップの硬いカットオフにフォールバックしつつコンソールに出力する。
            print(f"[MattingCanvas] estimate_alpha_cf failed: {e}")
            print(traceback.format_exc())
            self.alpha[y0:y1, x0:x1] = (trimap_crop.astype(np.float32) / 255.0)
            self.recompute_error = str(e)

        self._overlay_dirty = True
        self._preview_dirty = True

    def _recompute_full(self):
        if self.image_arr is None:
            return
        h, w = self.image_arr.shape[:2]
        self._recompute_region((0, 0, w, h))
        self.update()

    # ------------------------------------------------------------------
    # --- 座標変換 ---
    # ------------------------------------------------------------------
    def _widget_to_image(self, pos) -> tuple:
        x = (pos.x() - self.origin.x()) / self.zoom
        y = (pos.y() - self.origin.y()) / self.zoom
        return x, y

    # ------------------------------------------------------------------
    # --- ブラシ描画 ---
    # ------------------------------------------------------------------
    def _stamp_brush(self, ix: float, iy: float, bbox_acc: list):
        if self.trimap is None:
            return
        h, w = self.trimap.shape
        r = self.brush_size
        x0, x1 = int(max(0, ix - r)), int(min(w, ix + r + 1))
        y0, y1 = int(max(0, iy - r)), int(min(h, iy + r + 1))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - ix) ** 2 + (yy - iy) ** 2 <= r ** 2
        self.trimap[y0:y1, x0:x1][mask] = self.brush_mode

        bbox_acc[0] = min(bbox_acc[0], x0); bbox_acc[1] = min(bbox_acc[1], y0)
        bbox_acc[2] = max(bbox_acc[2], x1); bbox_acc[3] = max(bbox_acc[3], y1)

    def _stamp_line(self, p0, p1, bbox_acc: list):
        """速いドラッグでも隙間ができないよう、2点間を補間してスタンプする。"""
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        steps = max(1, int(dist / max(1, self.brush_size * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            self._stamp_brush(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t, bbox_acc)

    # ------------------------------------------------------------------
    # --- マウス/ホイール/キーボード イベント ---
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if self.image_arr is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._push_undo()
            self._drawing = True
            ix, iy = self._widget_to_image(event.position())
            self._last_img_pt = (ix, iy)
            self._stroke_bbox = [ix, iy, ix, iy]
            bbox_acc = self._stroke_bbox
            self._stamp_brush(ix, iy, bbox_acc)
            self._overlay_dirty = True
            self.update()
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._pan_anchor = (event.position(), QPointF(self.origin))

    def mouseMoveEvent(self, event):
        self._last_widget_pt = event.position()
        if self._drawing and self.image_arr is not None:
            ix, iy = self._widget_to_image(event.position())
            self._stamp_line(self._last_img_pt, (ix, iy), self._stroke_bbox)
            self._last_img_pt = (ix, iy)
            self._overlay_dirty = True
            self.update()
        elif self._panning and self._pan_anchor is not None:
            start_pos, start_origin = self._pan_anchor
            delta = event.position() - start_pos
            self.origin = QPointF(start_origin.x() + delta.x(), start_origin.y() + delta.y())
            self._user_adjusted_view = True
            self.update()
        else:
            self.update()  # ブラシカーソル追従のための再描画

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            if self._stroke_bbox is not None:
                x0, y0, x1, y1 = self._stroke_bbox
                bbox = (int(x0) - 1, int(y0) - 1, int(x1) + 1, int(y1) + 1)
                self._recompute_region(bbox)
            self._stroke_bbox = None
            self.update()
            self.status_changed.emit()
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self._pan_anchor = None

    def wheelEvent(self, event):
        # Ctrlキーが押されている場合はブラシサイズを変更、押されていない場合は既存のズーム処理
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                new_size = min(150, self.brush_size + 2)
            else:
                new_size = max(2, self.brush_size - 2)
            self.set_brush_size(new_size)
            event.accept()
            return

        if self.image_arr is None:
            return
        old_zoom = self.zoom
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = max(0.02, min(old_zoom * factor, 16.0))

        # カーソル位置を中心にズームする(カーソル直下の画像座標を維持)
        cursor = event.position()
        img_pt = self._widget_to_image(cursor)
        self.zoom = new_zoom
        self.origin = QPointF(
            cursor.x() - img_pt[0] * self.zoom,
            cursor.y() - img_pt[1] * self.zoom,
        )
        self._user_adjusted_view = True
        self.update()
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        ctrl = event.modifiers() & Qt.KeyboardModifier.ControlModifier
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if ctrl and event.key() == Qt.Key.Key_Z and shift:
            self.redo()
        elif ctrl and event.key() == Qt.Key.Key_Z:
            self.undo()
        elif ctrl and event.key() == Qt.Key.Key_Y:
            self.redo()
        elif event.key() == Qt.Key.Key_BracketLeft:
            new_size = max(2, self.brush_size - 5)
            self.set_brush_size(new_size)  # ここで自動的にシグナルが飛びます
            event.accept()
            return
        elif event.key() == Qt.Key.Key_BracketRight:
            new_size = min(150, self.brush_size + 5)
            self.set_brush_size(new_size)
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_arr is not None and not self._user_adjusted_view:
            self.fit_to_window()

    # ------------------------------------------------------------------
    # --- 描画本体 ---
    # ------------------------------------------------------------------
    def _rebuild_base_qimage(self):
        h, w = self.image_arr.shape[:2]
        self._base_qimage = QImage(self.image_arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)

    def _rebuild_overlay_qimage(self):
        h, w = self.trimap.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[self.trimap == self.BRUSH_UNKNOWN] = (255, 210, 0, 90)   # 不明領域: 黄色半透明
        overlay[self.trimap == self.BRUSH_BG] = (230, 30, 30, 60)        # 確実に背景: 赤系半透明
        self._overlay_qimage = QImage(overlay.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        self._overlay_dirty = False

    def _rebuild_preview_qimage(self):
        h, w = self.foreground_arr.shape[:2]
        # チェッカーボード背景に、現在のアルファで合成したプレビューを重ねる
        tile = 12
        checker = (((np.arange(h)[:, None] // tile) + (np.arange(w)[None, :] // tile)) % 2)
        bg = np.where(checker[..., None] == 0, 200, 150).astype(np.uint8)
        bg = np.repeat(bg, 3, axis=2)
        a = self.alpha[..., None]
        composed = (self.foreground_arr.astype(np.float32) * a + bg.astype(np.float32) * (1 - a))
        composed = np.clip(composed, 0, 255).astype(np.uint8)
        composed = np.ascontiguousarray(composed)
        self._preview_qimage = QImage(composed.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._preview_dirty = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(43, 43, 43))

        if self.image_arr is None:
            painter.setPen(QColor(190, 190, 190))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.property("empty_text") or "")
            painter.end()
            return

        if self.preview_alpha:
            if self._preview_dirty or self._preview_qimage is None:
                self._rebuild_preview_qimage()
            base_img = self._preview_qimage
        else:
            if self._base_qimage is None:
                self._rebuild_base_qimage()
            base_img = self._base_qimage

        painter.save()
        painter.translate(self.origin)
        painter.scale(self.zoom, self.zoom)
        painter.drawImage(0, 0, base_img)

        if not self.preview_alpha:
            if self._overlay_dirty or self._overlay_qimage is None:
                self._rebuild_overlay_qimage()
            painter.drawImage(0, 0, self._overlay_qimage)
        painter.restore()

        if self._last_widget_pt is not None and not self._panning:
            painter.setPen(QColor(255, 255, 255, 200))
            r = self.brush_size * self.zoom
            painter.drawEllipse(self._last_widget_pt, r, r)
        painter.end()

    # ------------------------------------------------------------------
    # --- 結果取得 ---
    # ------------------------------------------------------------------
    def get_result_rgba(self) -> Image.Image:
        h, w = self.foreground_arr.shape[:2]
        rgba = np.dstack([self.foreground_arr, np.clip(self.alpha * 255.0, 0, 255).astype(np.uint8)])
        return Image.fromarray(rgba, mode="RGBA")


# メインウィンドウクラス
class RembgGuiApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.resize(1300, 850)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(base_dir, "config.json")

        # --- i18n: UI構築前に保存済みの言語設定を読み込む ---
        # (ウィジェット生成時に初期テキストを正しい言語で出したいため、
        #  load_config() 本体より前にここだけ先読みする)
        self.i18n_registry = []  # [(widget, key, kind, kwargs), ...] retranslate_ui()用
        self.lang = self._load_language_preference()
        self.setWindowTitle(tr(self.lang, "window_title"))

        # --- ONNXモデル格納フォルダ(機能ごとに分離) ---
        self.onnx_rembg_dir = os.path.join(base_dir, "onnx", "rembg")
        self.onnx_upscale_dir = os.path.join(base_dir, "onnx", "upscale")

        # --- プリセット格納フォルダ(機能ごとに分離) ---
        self.presets_rembg_dir = os.path.join(base_dir, "presets", "rembg")
        self.presets_upscale_dir = os.path.join(base_dir, "presets", "upscale")
        self.presets_matting_dir = os.path.join(base_dir, "presets", "matting")

        # --- 出力先デフォルト(機能ごとに分離) ---
        self.default_output_rembg = os.path.join(base_dir, "output", "rembg")
        self.default_output_upscale = os.path.join(base_dir, "output", "upscale")
        self.default_output_matting = os.path.join(base_dir, "output", "matting")

        for d in (
            self.onnx_rembg_dir, self.onnx_upscale_dir,
            self.presets_rembg_dir, self.presets_upscale_dir, self.presets_matting_dir,
            self.default_output_rembg, self.default_output_upscale, self.default_output_matting,
        ):
            os.makedirs(d, exist_ok=True)

        self.init_ui()
        self._refresh_all_lists()
        self.load_config()
    
    def _refresh_all_lists(self):
        """起動時に一覧系ウィジェットを必ず最新化する。init_ui()末尾に書くと変更時に漏れやすいため独立させる。"""
        self.refresh_onnx_list()
        self.on_custom_toggle(None)
        self.refresh_preset_list()

    # ------------------------------------------------------------------
    # --- i18n(日本語/English) ヘルパー ---
    # ------------------------------------------------------------------
    def _load_language_preference(self) -> str:
        """config.json から言語設定だけを先読みする(UI構築より前に呼ぶ)。"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                lang = config.get("language", "ja")
                if lang in ("ja", "en"):
                    return lang
        except Exception as e:
            print(f"言語設定の読み込みエラー: {e}")
        return "ja"

    def tr(self, key: str, **kwargs) -> str:
        """現在のUI言語で文字列を取得する。"""
        return tr(self.lang, key, **kwargs)

    def _reg(self, widget, key: str, kind: str = "text", **kwargs):
        """ウィジェットをi18n登録し、現在の言語で初期テキストを設定する。
        kind: 'text'(setText) / 'title'(QGroupBox.setTitle) /
              'tooltip'(setToolTip) / 'placeholder'(setPlaceholderText)
        以後 retranslate_ui() を呼ぶだけで、登録した全ウィジェットの表示言語が切り替わる。
        """
        self.i18n_registry.append((widget, key, kind, kwargs))
        self._apply_i18n_item(widget, key, kind, kwargs)
        return widget

    def _apply_i18n_item(self, widget, key: str, kind: str, kwargs: dict):
        text = self.tr(key, **kwargs)
        if kind == "text":
            widget.setText(text)
        elif kind == "title":
            widget.setTitle(text)
        elif kind == "tooltip":
            widget.setToolTip(text)
        elif kind == "placeholder":
            widget.setPlaceholderText(text)

    def retranslate_ui(self):
        """言語切り替え時に呼ばれる。登録済みウィジェット + 個別処理が必要な要素を再翻訳する。"""
        self.setWindowTitle(self.tr("window_title"))

        for widget, key, kind, kwargs in self.i18n_registry:
            self._apply_i18n_item(widget, key, kind, kwargs)

        # タブ見出し(setTabTextは登録リストで扱いにくいため個別に)
        for index, key in enumerate(("tab_rembg", "tab_upscale", "tab_matting", "tab_output")):
            self.tab_widget.setTabText(index, self.tr(key))

        # 動的に文言が変わる要素は個別に再構築
        self.update_model_description(self.combo_model.currentText())
        self._retranslate_resample_combo()
        self._retranslate_output_dir_groups()
        if hasattr(self, "combo_upscale_standard"):
            self._update_upscale_model_status(self.combo_upscale_standard.currentText())
        if hasattr(self, "combo_matting_source"):
            self.combo_matting_source.setItemText(0, self.tr("matting_source_rembg"))
            self.combo_matting_source.setItemText(1, self.tr("matting_source_upscale"))
        if hasattr(self, "matting_canvas"):
            self.matting_canvas.setProperty("empty_text", self.tr("matting_canvas_empty"))
            self.matting_canvas.update()

        # プレビュー欄は「プレースホルダー文言」か「画像そのもの」かを状態に応じて出し分けているため、
        # setText() を無条件に呼ぶとpixmap表示中のプレビューが消えてしまう。
        # 現在の選択状態から表示を再計算させることで、両ケースとも正しい言語で描画し直す。
        self.update_image_preview(self.file_list.currentItem())

    def on_language_changed(self, index: int):
        new_lang = self.combo_language.itemData(index)
        if new_lang == self.lang:
            return
        self.lang = new_lang
        self.retranslate_ui()
        self.save_config()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- 左側: 共通パネル(ファイルリスト・プレビュー・ログ) ---
        # 「背景除去」「アップスケール」タブで共有。マットインペイントタブでは
        # 専用レイアウトを使うため、タブ切り替え時にこのウィジェットごと隠す。
        self.shared_left_widget = QWidget()
        self.shared_left_widget.setLayout(self._build_shared_left_panel())
        main_layout.addWidget(self.shared_left_widget, stretch=4)

        # --- 右側: タブウィジェット ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, stretch=3)

        matting_tab = self._build_matting_tab()
        self._matting_tab_widget = matting_tab

        self.tab_widget.addTab(self._build_rembg_tab(), self.tr("tab_rembg"))
        self.tab_widget.addTab(self._build_upscale_tab(), self.tr("tab_upscale"))
        self.tab_widget.addTab(matting_tab, self.tr("tab_matting"))
        self.tab_widget.addTab(self._build_output_settings_tab(), self.tr("tab_output"))

        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index: int):
        """マットインペイントタブは専用レイアウトのため、共有パネルの幅を0にして
        キャンバスに全幅を譲る(仕様: タブ切り替え時に比率を動的変更)。"""
        is_matting = self.tab_widget.widget(index) is getattr(self, "_matting_tab_widget", None)
        self.shared_left_widget.setVisible(not is_matting)
        if is_matting and hasattr(self, "matting_canvas") and self.matting_canvas.has_image():
            self.matting_canvas.update()

    def _build_shared_left_panel(self) -> QVBoxLayout:
        # ---------------- 左側: ファイルリスト ----------------
        left_layout = QVBoxLayout()
        drop_label = QLabel()
        self._reg(drop_label, "drop_files_label")
        left_layout.addWidget(drop_label)

        self.file_list = DropListWidget()
        left_layout.addWidget(self.file_list, stretch=2) # サイズ割合

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton()
        self._reg(self.btn_add, "btn_add_files")
        self.btn_add.clicked.connect(self.add_files_dialog)
        self.btn_remove = QPushButton()
        self._reg(self.btn_remove, "btn_remove_selected")
        self.btn_remove.clicked.connect(self.remove_selected_items)
        self.btn_clear = QPushButton()
        self._reg(self.btn_clear, "btn_clear_list")
        self.btn_clear.clicked.connect(self.file_list.clear)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_clear)
        left_layout.addLayout(btn_layout)

        # --- 変更: プレビューグループをここ(中段)に移動 ---
        preview_group = QGroupBox()
        self._reg(preview_group, "preview_group", kind="title")
        preview_layout = QHBoxLayout()

        self.label_preview_before = QLabel(self.tr("preview_before_empty"))
        self.label_preview_before.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview_before.setMinimumSize(150, 150)  # 左パネル幅に合わせて縮小
        self.label_preview_before.setStyleSheet("background-color: #2b2b2b; color: #ccc; border: 1px solid #555;")

        self.label_preview_after = QLabel(self.tr("preview_after_empty"))
        self.label_preview_after.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview_after.setMinimumSize(150, 150)
        self.label_preview_after.setStyleSheet("background-color: #444; color: #ccc; border: 1px solid #555;")

        preview_layout.addWidget(self.label_preview_before)
        preview_layout.addWidget(self.label_preview_after)
        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group, stretch=2)  # サイズ割合

        # --- プログレスバー ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # --- 変更: log_label(QLabel) → log_view(QTextEdit) でスクロール履歴表示に ---
        log_label = QLabel()
        self._reg(log_label, "log_label")
        left_layout.addWidget(log_label)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self._reg(self.log_view, "log_placeholder", kind="placeholder")
        left_layout.addWidget(self.log_view, stretch=1)  # サイズ割合

        self.file_list.currentItemChanged.connect(self.update_image_preview)
        return left_layout

    def _build_rembg_tab(self) -> QWidget:
        """既存の「自前ONNXモデル」「マスクエッジ調整」「スピル除去」「パフォーマンス」
        「プリセット」「実行/キャンセルボタン」の各グループボックスをまとめて1つのタブにする"""
        tab = QWidget()
        tab.setStyleSheet("QWidget { font-size: 10pt; } QGroupBox { font-size: 10pt; font-weight; bold; }")
        right_layout = QVBoxLayout(tab)

        # ---------------- 右側: 設定オプション（省スペース設計） ----------------
        right_layout.setSpacing(5)  # 各ウィジェット間の隙間を詰める

        # --- プリセットグループ ---
        preset_group = QGroupBox()
        self._reg(preset_group, "preset_group", kind="title")
        preset_form = QFormLayout()
        preset_form.setContentsMargins(5, 5, 5, 5)

        self.combo_preset = QComboBox()
        self.combo_preset.setEditable(True)
        self.combo_preset.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        preset_form.addRow(self._reg(QLabel(), "preset_name_label"), self.combo_preset)

        preset_btn_layout = QHBoxLayout()
        self.btn_save_preset = QPushButton()
        self._reg(self.btn_save_preset, "btn_save")
        self.btn_save_preset.clicked.connect(self.save_preset)
        self.btn_load_preset = QPushButton()
        self._reg(self.btn_load_preset, "btn_load")
        self.btn_load_preset.clicked.connect(self.load_preset)
        self.btn_refresh_preset = QPushButton()
        self._reg(self.btn_refresh_preset, "btn_refresh")
        self.btn_refresh_preset.clicked.connect(self.refresh_preset_list)

        preset_btn_layout.addWidget(self.btn_save_preset)
        preset_btn_layout.addWidget(self.btn_load_preset)
        preset_btn_layout.addWidget(self.btn_refresh_preset)
        preset_form.addRow(preset_btn_layout)

        preset_group.setLayout(preset_form)
        right_layout.addWidget(preset_group)

        # --- 1. 自前ONNXモデル選択グループ ---
        custom_group = QGroupBox()
        self._reg(custom_group, "custom_onnx_group", kind="title")
        custom_form = QFormLayout()
        custom_form.setContentsMargins(3, 3, 3, 3)
        custom_form.setVerticalSpacing(2)

        # --- パフォーマンス設定 ---

        perf_group = QGroupBox()
        self._reg(perf_group, "perf_group", kind="title")
        perf_form = QFormLayout()
        perf_form.setContentsMargins(5, 5, 5, 5)

        self.check_use_batch = QCheckBox()
        self._reg(self.check_use_batch, "check_use_batch")
        self.check_use_batch.setChecked(False)
        self.check_use_batch.setToolTip(
            self.tr("check_use_batch") + "\n" +
            ("複数枚をまとめてGPUに投げることで高速化しますが、rembgのアルファマッティング後処理は適用されなくなります。"
             if self.lang == "ja" else
             "Sends multiple images to the GPU at once for speed, but rembg's alpha-matting post-processing will not be applied.")
        )
        self.check_use_batch.stateChanged.connect(self.on_batch_toggle)
        perf_form.addRow(self.check_use_batch)

        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(2, 32)
        self.spin_batch_size.setValue(4)
        self.spin_batch_size.setEnabled(False)  # 初期はチェックボックスOFFなので無効
        self.spin_batch_size.setToolTip(
            "まとめてGPUに投げる画像数。大きいほど高速だがVRAM消費も増える。" if self.lang == "ja"
            else "Number of images sent to the GPU at once. Larger = faster but uses more VRAM."
        )
        perf_form.addRow(self._reg(QLabel(), "batch_size_label"), self.spin_batch_size)

        perf_group.setLayout(perf_form)
        right_layout.addWidget(perf_group)

        # 自前/標準切り替えチェックボックス
        self.check_use_custom = QCheckBox()
        self._reg(self.check_use_custom, "check_use_custom")
        self.check_use_custom.setChecked(False)
        self.check_use_custom.stateChanged.connect(self.on_custom_toggle)
        custom_form.addRow(self.check_use_custom)

        # ドロップダウン + 更新ボタン
        self.combo_custom_onnx = QComboBox()
        self.btn_refresh_onnx = QPushButton()
        self._reg(self.btn_refresh_onnx, "btn_refresh")
        self.btn_refresh_onnx.setToolTip(
            f"{self.onnx_rembg_dir} " + ("内の.onnxファイルを再スキャンします" if self.lang == "ja" else "— rescan .onnx files in this folder")
        )
        self.btn_refresh_onnx.clicked.connect(self.refresh_onnx_list)

        onnx_combo_layout = QHBoxLayout()
        onnx_combo_layout.addWidget(self.combo_custom_onnx)
        onnx_combo_layout.addWidget(self.btn_refresh_onnx)
        custom_form.addRow(self._reg(QLabel(), "model_select_label"), onnx_combo_layout)

        # 外部ファイルを選択 → onnx/ フォルダへ自動コピーして登録
        self.btn_browse_onnx = QPushButton()
        self._reg(self.btn_browse_onnx, "btn_browse_onnx")
        self.btn_browse_onnx.setToolTip(
            (f"選択したファイルを {self.onnx_rembg_dir} にコピーして一覧に追加します") if self.lang == "ja"
            else (f"Copies the selected file into {self.onnx_rembg_dir} and adds it to the list")
        )
        self.btn_browse_onnx.clicked.connect(self.browse_onnx_file)
        custom_form.addRow(self.btn_browse_onnx)

        custom_group.setLayout(custom_form)
        right_layout.addWidget(custom_group)
        
        # --- 2. 標準AIモデル選択グループ ---
        model_group = QGroupBox()
        self._reg(model_group, "model_group", kind="title")
        model_form = QFormLayout()
        model_form.setContentsMargins(5, 5, 5, 5)
        
        self.combo_model = QComboBox()
        models = [
            "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta",
            "isnet-general-use", "isnet-anime", "sam",
            "birefnet-general", "birefnet-general-lite", "birefnet-portrait",
            "birefnet-dis", "birefnet-hrsod", "birefnet-cod", "birefnet-massive",
            "bria-rmbg"
        ]
        self.combo_model.addItems(models)
        self.combo_model.setCurrentText("u2net")
        
        self.model_desc = QLabel()
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet("color: #b5b5b5; font-size: 11px;")
        self.combo_model.currentTextChanged.connect(self.update_model_description)

        model_form.addRow(self._reg(QLabel(), "model_label"), self.combo_model)
        model_form.addRow(self.model_desc)
        model_group.setLayout(model_form)
        right_layout.addWidget(model_group)
        self.update_model_description(self.combo_model.currentText())  # 初期説明文をセット

        # --- リサンプリンググループ ---
        resample_group = QGroupBox()
        self._reg(resample_group, "resample_group", kind="title")
        resample_form = QFormLayout()
        resample_form.setContentsMargins(5, 5, 5, 5)

        # 内部キー(bilinear/bicubic/lanczos3)をuserDataとして保持し、
        # 表示ラベルだけを言語ごとに切り替える(プリセット互換性・言語非依存の保存のため)。
        self.combo_resample = QComboBox()
        for key in RESAMPLE_KEYS:
            self.combo_resample.addItem(self.tr(f"resample_{key}"), key)
        self.combo_resample.setCurrentIndex(self.combo_resample.findData("lanczos3"))  # 既定値
        resample_form.addRow(self._reg(QLabel(), "resample_label"), self.combo_resample)

        resample_group.setLayout(resample_form)
        right_layout.addWidget(resample_group)
        self._retranslate_resample_combo(initial=True)

        # --- 3. マスクのダイレクト調整グループ ---
        mask_adjust_group = QGroupBox()
        self._reg(mask_adjust_group, "mask_adjust_group", kind="title")
        mask_form = QFormLayout()
        mask_form.setContentsMargins(5, 5, 5, 5)

        self.spin_m_blur = QSpinBox()
        self.spin_m_blur.setRange(0, 50)
        self.spin_m_blur.setValue(0)
        mask_form.addRow(self._reg(QLabel(), "mask_blur_label"), self.spin_m_blur)

        self.spin_m_thresh = QSpinBox()
        self.spin_m_thresh.setRange(-255, 255)  # ← マイナス値を受け付ける
        self.spin_m_thresh.setValue(0)          # 初期値は 0
        mask_form.addRow(self._reg(QLabel(), "mask_offset_label"), self.spin_m_thresh)

        mask_adjust_group.setLayout(mask_form)
        right_layout.addWidget(mask_adjust_group)

        # --- 3.5 色スピル除去グループ ---
        spill_group = QGroupBox()
        self._reg(spill_group, "spill_group", kind="title")
        spill_form = QFormLayout()
        spill_form.setContentsMargins(5, 5, 5, 5)

        self.check_spill = QCheckBox()
        self._reg(self.check_spill, "check_enable")
        self.check_spill.setChecked(True)
        spill_form.addRow(self._reg(QLabel(), "spill_enable_label"), self.check_spill)

        self.spin_spill_threshold = QSpinBox()
        self.spin_spill_threshold.setRange(0, 255)
        self.spin_spill_threshold.setValue(60)
        self.spin_spill_threshold.setToolTip(
            "大きいほど背景色に近い色まで広く透明化します" if self.lang == "ja"
            else "Larger values make colors closer to the background more transparent"
        )
        spill_form.addRow(self._reg(QLabel(), "spill_threshold_label"), self.spin_spill_threshold)

        self.spin_spill_patch = QSpinBox()
        self.spin_spill_patch.setRange(1, 100)
        self.spin_spill_patch.setValue(10)
        self.spin_spill_patch.setToolTip(
            "背景色を推定するために画像四隅から取るサンプル領域の大きさ(px)" if self.lang == "ja"
            else "Size (px) of the corner sample patches used to estimate the background color"
        )
        spill_form.addRow(self._reg(QLabel(), "spill_patch_label"), self.spin_spill_patch)

        spill_group.setLayout(spill_form)
        right_layout.addWidget(spill_group)

        # --- 自動穴埋め処理グループ ---
        hole_fill_group = QGroupBox()
        self._reg(hole_fill_group, "hole_fill_group", kind="title")
        hole_fill_form = QFormLayout()
        hole_fill_form.setContentsMargins(5, 5, 5, 5)

        self.check_hole_fill = QCheckBox()
        self._reg(self.check_hole_fill, "check_enable")
        self.check_hole_fill.setChecked(True)
        hole_fill_form.addRow(self._reg(QLabel(), "hole_fill_label"), self.check_hole_fill)

        self.spin_hole_fill_threshold = QSpinBox()
        self.spin_hole_fill_threshold.setRange(1, 254)
        self.spin_hole_fill_threshold.setValue(127)
        self.spin_hole_fill_threshold.setToolTip(
            "この値以上を「前景」とみなして穴の判定を行う" if self.lang == "ja"
            else "Pixels at or above this value are treated as \"foreground\" when detecting holes"
        )
        hole_fill_form.addRow(self._reg(QLabel(), "hole_fill_threshold_label"), self.spin_hole_fill_threshold)

        self.spin_hole_fill_max_area = QSpinBox()
        self.spin_hole_fill_max_area.setRange(0, 100000)
        self.spin_hole_fill_max_area.setValue(500)
        self.spin_hole_fill_max_area.setToolTip(
            ("この面積(px²)を超える穴は、服の隙間等の意図的な透過部分とみなし埋めない。0にすると大きさを問わず全ての孤立穴を埋める。")
            if self.lang == "ja" else
            ("Holes larger than this area (px²) are treated as intentional gaps (e.g. between clothes) and left as-is. "
             "0 fills every isolated hole regardless of size.")
        )
        hole_fill_form.addRow(self._reg(QLabel(), "hole_fill_max_area_label"), self.spin_hole_fill_max_area)

        hole_fill_group.setLayout(hole_fill_form)
        right_layout.addWidget(hole_fill_group)

        # --- 4. 詳細設定グループ（アルファマッティング） ---
        option_group = QGroupBox()
        self._reg(option_group, "alpha_matting_group", kind="title")
        form_layout = QFormLayout()
        form_layout.setContentsMargins(5, 5, 5, 5)

        self.check_om = QCheckBox()
        self._reg(self.check_om, "check_om")
        self.check_om.setChecked(False)
        form_layout.addRow(self._reg(QLabel(), "only_mask_label"), self.check_om)

        self.check_am = QCheckBox()
        self._reg(self.check_am, "check_enable")
        self.check_am.setChecked(True)
        form_layout.addRow(self._reg(QLabel(), "alpha_matting_label"), self.check_am)

        self.spin_bg = QSpinBox()
        self.spin_bg.setRange(0, 255)
        self.spin_bg.setValue(10)
        form_layout.addRow(self._reg(QLabel(), "bg_threshold_label"), self.spin_bg)

        self.spin_fg = QSpinBox()
        self.spin_fg.setRange(0, 255)
        self.spin_fg.setValue(240)
        form_layout.addRow(self._reg(QLabel(), "fg_threshold_label"), self.spin_fg)

        self.spin_erode = QSpinBox()
        self.spin_erode.setRange(0, 100)
        self.spin_erode.setValue(10)
        form_layout.addRow(self._reg(QLabel(), "erode_size_label"), self.spin_erode)

        option_group.setLayout(form_layout)
        right_layout.addWidget(option_group)

        # 実行ボタン + キャンセルボタン
        right_layout.addSpacing(5)
        btn_run_layout = QHBoxLayout()

        self.btn_run = QPushButton()
        self._reg(self.btn_run, "btn_run")
        self.btn_run.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #2b8a3e; color: white; padding: 8px;")
        self.btn_run.clicked.connect(self.start_batch_processing)

        self.btn_cancel = QPushButton()
        self._reg(self.btn_cancel, "btn_cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #c92a2a; color: white; padding: 8px;")
        self.btn_cancel.clicked.connect(self.cancel_batch_processing)

        btn_run_layout.addWidget(self.btn_run)
        btn_run_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(btn_run_layout)

        right_layout.addStretch(1)
        return tab

    # ------------------------------------------------------------------
    # --- アップスケールタブ ---
    # ------------------------------------------------------------------
    def _build_upscale_tab(self) -> QWidget:
        tab = QWidget()
        right_layout = QVBoxLayout(tab)
        right_layout.setSpacing(5)

        # --- 標準モデル(HFから自動DL) ---
        model_group = QGroupBox()
        self._reg(model_group, "upscale_model_group", kind="title")
        model_form = QFormLayout()
        model_form.setContentsMargins(5, 5, 5, 5)

        self.combo_upscale_standard = QComboBox()
        self.combo_upscale_standard.addItems(list(UPSCALE_STANDARD_MODELS.keys()))
        self.combo_upscale_standard.currentTextChanged.connect(self._update_upscale_model_status)
        model_form.addRow(self._reg(QLabel(), "upscale_standard_label"), self.combo_upscale_standard)

        self.label_upscale_model_status = QLabel()
        self.label_upscale_model_status.setStyleSheet("color: #b5b5b5; font-size: 11px;")
        model_form.addRow(self.label_upscale_model_status)

        self.btn_download_upscale_model = QPushButton()
        self._reg(self.btn_download_upscale_model, "btn_download_model")
        self.btn_download_upscale_model.clicked.connect(self.download_upscale_model)
        model_form.addRow(self.btn_download_upscale_model)

        model_group.setLayout(model_form)
        right_layout.addWidget(model_group)

        # --- 自前アップスケールモデル(4x-UltraSharp等、非商用ライセンス注意) ---
        custom_group = QGroupBox()
        self._reg(custom_group, "upscale_custom_group", kind="title")
        custom_form = QFormLayout()
        custom_form.setContentsMargins(5, 5, 5, 5)

        self.check_use_custom_upscale = QCheckBox()
        self._reg(self.check_use_custom_upscale, "check_use_custom")
        self.check_use_custom_upscale.stateChanged.connect(self.on_custom_upscale_toggle)
        custom_form.addRow(self.check_use_custom_upscale)

        self.combo_custom_upscale = QComboBox()
        self.combo_custom_upscale.setEnabled(False)
        self.btn_refresh_upscale_onnx = QPushButton()
        self._reg(self.btn_refresh_upscale_onnx, "btn_refresh")
        self.btn_refresh_upscale_onnx.clicked.connect(self.refresh_upscale_onnx_list)
        self.btn_refresh_upscale_onnx.setEnabled(False)

        onnx_row = QHBoxLayout()
        onnx_row.addWidget(self.combo_custom_upscale)
        onnx_row.addWidget(self.btn_refresh_upscale_onnx)
        custom_form.addRow(self._reg(QLabel(), "model_select_label"), onnx_row)

        self.btn_browse_upscale_onnx = QPushButton()
        self._reg(self.btn_browse_upscale_onnx, "btn_browse_onnx")
        self.btn_browse_upscale_onnx.setEnabled(False)
        self.btn_browse_upscale_onnx.clicked.connect(self.browse_upscale_onnx_file)
        custom_form.addRow(self.btn_browse_upscale_onnx)

        note = QLabel()
        self._reg(note, "upscale_noncommercial_note")
        note.setWordWrap(True)
        note.setStyleSheet("color: #e0a030; font-size: 11px;")
        custom_form.addRow(note)

        custom_group.setLayout(custom_form)
        right_layout.addWidget(custom_group)

        # --- タイル分割設定(VRAM対策) ---
        tile_group = QGroupBox()
        self._reg(tile_group, "upscale_tile_group", kind="title")
        tile_form = QFormLayout()
        tile_form.setContentsMargins(5, 5, 5, 5)

        self.spin_tile_size = QSpinBox()
        self.spin_tile_size.setRange(128, 1024)
        self.spin_tile_size.setSingleStep(32)
        self.spin_tile_size.setValue(512)
        self.spin_tile_size.setToolTip(
            "大きいほど高速・高精度になりますがVRAM消費が急増します(RTX4070Ti 12GBなら512が目安)。"
            if self.lang == "ja" else
            "Larger = faster and higher quality but VRAM usage rises sharply (512 is a good default on a 12GB RTX 4070 Ti)."
        )
        tile_form.addRow(self._reg(QLabel(), "upscale_tile_size_label"), self.spin_tile_size)

        self.spin_tile_overlap = QSpinBox()
        self.spin_tile_overlap.setRange(0, 128)
        self.spin_tile_overlap.setSingleStep(8)
        self.spin_tile_overlap.setValue(32)
        self.spin_tile_overlap.setToolTip(
            "タイル境界のブレンド幅。大きいほど継ぎ目は目立たなくなりますが計算量が増えます。"
            if self.lang == "ja" else
            "Blend width at tile boundaries. Larger reduces visible seams but costs more compute."
        )
        tile_form.addRow(self._reg(QLabel(), "upscale_overlap_label"), self.spin_tile_overlap)

        self.label_tile_fixed_note = QLabel()
        self._reg(self.label_tile_fixed_note, "upscale_tile_fixed_note")
        self.label_tile_fixed_note.setWordWrap(True)
        self.label_tile_fixed_note.setStyleSheet("color: #888; font-size: 10px;")
        self.label_tile_fixed_note.setVisible(False)
        tile_form.addRow(self.label_tile_fixed_note)

        tile_group.setLayout(tile_form)
        right_layout.addWidget(tile_group)

        # --- 出力倍率の微調整(任意、既存のresampleコンボを再利用) ---
        scale_group = QGroupBox()
        self._reg(scale_group, "upscale_target_scale_group", kind="title")
        scale_form = QFormLayout()
        scale_form.setContentsMargins(5, 5, 5, 5)

        self.spin_target_scale = QDoubleSpinBox()
        self.spin_target_scale.setRange(0.0, 8.0)
        self.spin_target_scale.setSingleStep(0.5)
        self.spin_target_scale.setValue(0.0)
        self.spin_target_scale.setToolTip(
            "0のままならモデル本来の倍率(x2/x4等)で出力します。値を指定すると\n"
            "モデル出力後に「リサンプリング方式」タブの設定を使ってその倍率へ再調整します。"
            if self.lang == "ja" else
            "Leave at 0 to output at the model's native scale (x2/x4, etc.). If set, the model's "
            "output is rescaled to this factor afterward using the Resampling Method setting."
        )
        scale_form.addRow(self._reg(QLabel(), "upscale_target_scale_label"), self.spin_target_scale)
        scale_group.setLayout(scale_form)
        right_layout.addWidget(scale_group)

        # --- 実行/キャンセル(共有の進捗バー・ログを流用) ---
        run_group = QGroupBox()
        self._reg(run_group, "upscale_run_group", kind="title")
        run_layout = QHBoxLayout()

        self.btn_run_upscale = QPushButton()
        self._reg(self.btn_run_upscale, "btn_run")
        self.btn_run_upscale.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #2b8a3e; color: white; padding: 8px;")
        self.btn_run_upscale.clicked.connect(self.start_upscale_processing)

        self.btn_cancel_upscale = QPushButton()
        self._reg(self.btn_cancel_upscale, "btn_cancel")
        self.btn_cancel_upscale.setEnabled(False)
        self.btn_cancel_upscale.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #c92a2a; color: white; padding: 8px;")
        self.btn_cancel_upscale.clicked.connect(self.cancel_upscale_processing)

        run_layout.addWidget(self.btn_run_upscale)
        run_layout.addWidget(self.btn_cancel_upscale)
        run_group.setLayout(run_layout)
        right_layout.addWidget(run_group)

        right_layout.addStretch(1)
        self._update_upscale_model_status(self.combo_upscale_standard.currentText())
        return tab

    def _standard_upscale_model_path(self, model_key: str) -> str:
        info = UPSCALE_STANDARD_MODELS[model_key]
        return os.path.join(self.onnx_upscale_dir, info["filename"])

    def _update_upscale_model_status(self, model_key: str):
        if not model_key:
            return
        path = self._standard_upscale_model_path(model_key)
        ready = os.path.exists(path)
        self.label_upscale_model_status.setText(
            self.tr("upscale_model_ready") if ready else self.tr("upscale_model_missing")
        )
        self.btn_download_upscale_model.setEnabled(not ready)

    def download_upscale_model(self):
        model_key = self.combo_upscale_standard.currentText()
        info = UPSCALE_STANDARD_MODELS[model_key]
        reply = QMessageBox.question(
            self, self.tr("confirm_title"),
            self.tr("msg_download_confirm", name=model_key, repo=info["repo_id"]),
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_download_upscale_model.setEnabled(False)
        self.dl_thread = ModelDownloadThread(
            info["repo_id"], info["filename"], self.onnx_upscale_dir, lang=self.lang
        )
        self.dl_thread.progress_signal.connect(self.progress_bar.setValue)
        self.dl_thread.log_signal.connect(self.append_log)
        self.dl_thread.finished_signal.connect(self._on_download_finished)
        self.dl_thread.start()

    def _on_download_finished(self, success: bool, message: str):
        if not success:
            self.append_log(self.tr("download_failed", err=message))
            QMessageBox.warning(self, self.tr("error_title"), self.tr("download_failed", err=message))
        self._update_upscale_model_status(self.combo_upscale_standard.currentText())
        self.progress_bar.setValue(0)

    def refresh_upscale_onnx_list(self):
        self.combo_custom_upscale.clear()
        files = sorted(f for f in os.listdir(self.onnx_upscale_dir) if f.lower().endswith(".onnx"))
        if not files:
            self.combo_custom_upscale.addItem(self.tr("onnx_list_empty"))
        else:
            self.combo_custom_upscale.addItems(files)

    def on_custom_upscale_toggle(self, state):
        is_custom = self.check_use_custom_upscale.isChecked()
        self.combo_custom_upscale.setEnabled(is_custom)
        self.btn_refresh_upscale_onnx.setEnabled(is_custom)
        self.btn_browse_upscale_onnx.setEnabled(is_custom)
        self.combo_upscale_standard.setEnabled(not is_custom)
        self.btn_download_upscale_model.setEnabled(not is_custom and self.btn_download_upscale_model.isEnabled())
        if is_custom:
            self.refresh_upscale_onnx_list()

    def browse_upscale_onnx_file(self):
        dialog_title = "自前アップスケールモデルを選択" if self.lang == "ja" else "Select Custom Upscale Model"
        file_path, _ = QFileDialog.getOpenFileName(self, dialog_title, "", "ONNX Models (*.onnx)")
        if not file_path:
            return
        dest_path = os.path.join(self.onnx_upscale_dir, os.path.basename(file_path))
        try:
            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                shutil.copy2(file_path, dest_path)
            self.refresh_upscale_onnx_list()
            self.combo_custom_upscale.setCurrentText(os.path.basename(dest_path))
        except Exception as e:
            QMessageBox.warning(self, self.tr("error_title"), self.tr("msg_onnx_copy_failed", err=e))

    def start_upscale_processing(self):
        file_count = self.file_list.count()
        if file_count == 0:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_files"))
            return

        output_dir = self.edit_output_upscale.text().strip()
        if not output_dir:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_output_dir"))
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_output_dir_not_found"))
            return

        if self.check_use_custom_upscale.isChecked():
            selected_file = self.combo_custom_upscale.currentText()
            if not selected_file or not selected_file.lower().endswith(".onnx"):
                QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_custom_onnx_selected"))
                return
            model_path = os.path.join(self.onnx_upscale_dir, selected_file)
        else:
            model_key = self.combo_upscale_standard.currentText()
            model_path = self._standard_upscale_model_path(model_key)
            if not os.path.exists(model_path):
                QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_model_not_ready"))
                return

        file_paths = [self.file_list.item(i).text() for i in range(file_count)]
        target_scale = self.spin_target_scale.value() or None
        resample_method = RESAMPLE_PIL[self.combo_resample.currentData()]

        self.upscale_thread = UpscaleProcessThread(
            file_paths, output_dir, model_path,
            tile_size=self.spin_tile_size.value(),
            overlap=self.spin_tile_overlap.value(),
            target_scale=target_scale,
            resample_method=resample_method,
            lang=self.lang,
        )
        self.upscale_thread.progress_signal.connect(self.progress_bar.setValue)
        self.upscale_thread.log_signal.connect(self.append_log)
        self.upscale_thread.finished_signal.connect(self.on_upscale_finished)

        self.btn_run_upscale.setEnabled(False)
        self.btn_cancel_upscale.setEnabled(True)
        self.upscale_thread.start()

    def cancel_upscale_processing(self):
        if hasattr(self, "upscale_thread") and self.upscale_thread.isRunning():
            self.upscale_thread.cancel()
        self.btn_cancel_upscale.setEnabled(False)
        self.append_log(self.tr("log_cancel_requested"))

    def on_upscale_finished(self, success_count):
        self.btn_run_upscale.setEnabled(True)
        self.btn_cancel_upscale.setEnabled(False)
        self.append_log(log_t(self.lang, "batch_complete", count=success_count))

    # ------------------------------------------------------------------
    # --- マットインペイントタブ ---
    # 仕様上、共有パネル(file_list/プレビュー/ログ)は使わずタブ内で完結させる。
    # ------------------------------------------------------------------
    def _build_matting_tab(self) -> QWidget:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        # ---------------- 左: 出力元切り替え + タブ内専用ファイル一覧 ----------------
        left = QWidget()
        left.setMaximumWidth(260)
        left_layout = QVBoxLayout(left)

        self.combo_matting_source = QComboBox()
        self.combo_matting_source.addItem(self.tr("matting_source_rembg"), "rembg")
        self.combo_matting_source.addItem(self.tr("matting_source_upscale"), "upscale")
        self.combo_matting_source.currentIndexChanged.connect(self.refresh_matting_file_list)
        left_layout.addWidget(self._reg(QLabel(), "matting_source_label"))
        left_layout.addWidget(self.combo_matting_source)

        self.matting_file_list = QListWidget()
        self.matting_file_list.currentItemChanged.connect(self.on_matting_file_selected)
        left_layout.addWidget(self.matting_file_list, stretch=1)

        if not PYMATTING_AVAILABLE:
            warn = QLabel()
            self._reg(warn, "matting_no_pymatting")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #e05050; font-size: 11px;")
            left_layout.addWidget(warn)

        # 修正: 共有ログ(log_view)はこのタブでは非表示になるため、
        # 保存結果や再計算エラーをこのタブ内だけで完結して確認できるよう専用ラベルを設置する。
        self.label_matting_status = QLabel()
        self.label_matting_status.setWordWrap(True)
        self.label_matting_status.setStyleSheet("color: #b5b5b5; font-size: 11px;")
        left_layout.addWidget(self.label_matting_status)

        # --- ブラシ設定 ---
        brush_group = QGroupBox()
        self._reg(brush_group, "matting_brush_group", kind="title")
        brush_form = QFormLayout()
        brush_form.setContentsMargins(5, 5, 5, 5)

        self.slider_brush_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_size.setRange(2, 150)
        self.slider_brush_size.setValue(30)
        self.slider_brush_size.valueChanged.connect(lambda v: self.matting_canvas.set_brush_size(v))
        brush_form.addRow(self._reg(QLabel(), "matting_brush_size_label"), self.slider_brush_size)

        self.brush_mode_group = QButtonGroup(self)
        radio_unknown = QRadioButton()
        self._reg(radio_unknown, "matting_brush_unknown")
        radio_unknown.setChecked(True)
        radio_fg = QRadioButton()
        self._reg(radio_fg, "matting_brush_fg")
        radio_bg = QRadioButton()
        self._reg(radio_bg, "matting_brush_bg")
        for rb, mode in ((radio_unknown, MattingCanvas.BRUSH_UNKNOWN),
                         (radio_fg, MattingCanvas.BRUSH_FG),
                         (radio_bg, MattingCanvas.BRUSH_BG)):
            self.brush_mode_group.addButton(rb)
            rb.toggled.connect(lambda checked, m=mode: self.matting_canvas.set_brush_mode(m) if checked else None)
            brush_form.addRow(rb)

        brush_group.setLayout(brush_form)
        left_layout.addWidget(brush_group)

        # --- Undo/Redo/表示調整 ---
        history_layout = QHBoxLayout()
        self.btn_matting_undo = QPushButton()
        self._reg(self.btn_matting_undo, "btn_undo")
        self.btn_matting_undo.clicked.connect(lambda: self.matting_canvas.undo())
        self.btn_matting_redo = QPushButton()
        self._reg(self.btn_matting_redo, "btn_redo")
        self.btn_matting_redo.clicked.connect(lambda: self.matting_canvas.redo())
        history_layout.addWidget(self.btn_matting_undo)
        history_layout.addWidget(self.btn_matting_redo)
        left_layout.addLayout(history_layout)

        self.btn_matting_fit = QPushButton()
        self._reg(self.btn_matting_fit, "btn_fit_view")
        self.btn_matting_fit.clicked.connect(lambda: self.matting_canvas.fit_to_window())
        left_layout.addWidget(self.btn_matting_fit)

        self.check_preview_alpha = QCheckBox()
        self._reg(self.check_preview_alpha, "check_preview_alpha")
        self.check_preview_alpha.stateChanged.connect(
            lambda s: self.matting_canvas.set_preview_alpha(self.check_preview_alpha.isChecked())
        )
        left_layout.addWidget(self.check_preview_alpha)

        self.btn_save_matting = QPushButton()
        self._reg(self.btn_save_matting, "btn_save_matting")
        self.btn_save_matting.setStyleSheet("font-weight: bold; background-color: #2b8a3e; color: white; padding: 6px;")
        self.btn_save_matting.clicked.connect(self.save_matting_result)
        left_layout.addWidget(self.btn_save_matting)

        outer.addWidget(left)

        # ---------------- 右: 編集キャンバス ----------------
        self.matting_canvas = MattingCanvas()
        self.matting_canvas.setProperty("empty_text", self.tr("matting_canvas_empty"))
        self.matting_canvas.status_changed.connect(self._update_matting_history_buttons)
        outer.addWidget(self.matting_canvas, stretch=1)

        self._update_matting_history_buttons()

        if hasattr(self, "matting_canvas") and hasattr(self, "slider_brush_size"):
            self.matting_canvas.brush_size_changed.connect(self.slider_brush_size.setValue)
        return tab

    def refresh_matting_file_list(self):
        self.matting_file_list.clear()
        source = self.combo_matting_source.currentData()
        if source == "upscale":
            src_dir = self.edit_output_upscale.text().strip()
        else:
            src_dir = self.edit_output_rembg.text().strip()
        if not src_dir or not os.path.isdir(src_dir):
            return
        files = sorted(f for f in os.listdir(src_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")))
        for f in files:
            self.matting_file_list.addItem(os.path.join(src_dir, f))

    def on_matting_file_selected(self, current, previous=None):
        if current is None:
            return
        try:
            img = Image.open(current.text())
            self.matting_canvas.load_image(img)
        except Exception as e:
            self.append_log(self.tr("log_matting_recompute_failed", err=str(e)))
        self._update_matting_history_buttons()

    def _update_matting_history_buttons(self):
        if not hasattr(self, "matting_canvas"):
            return
        self.btn_matting_undo.setEnabled(self.matting_canvas.can_undo())
        self.btn_matting_redo.setEnabled(self.matting_canvas.can_redo())
        err = self.matting_canvas.recompute_error
        if err:
            if err == "pymatting_missing":
                msg = self.tr("matting_no_pymatting")
            else:
                msg = self.tr("log_matting_recompute_failed", err=err)
            self._set_matting_status(msg, is_error=True)
            self.append_log(msg)  # 共有ログにも残す(このタブでは非表示だが履歴としては保持)
            self.matting_canvas.recompute_error = None

    def _set_matting_status(self, text: str, is_error: bool = False):
        if not hasattr(self, "label_matting_status"):
            return
        self.label_matting_status.setText(text)
        self.label_matting_status.setStyleSheet(
            "color: #e05050; font-size: 11px;" if is_error else "color: #6fbf6f; font-size: 11px;"
        )

    def save_matting_result(self):
        if not self.matting_canvas.has_image():
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_matting_image"))
            return
        current_item = self.matting_file_list.currentItem()
        stem = Path(current_item.text()).stem if current_item else "matting_result"
        out_dir = self.edit_output_matting.text().strip()
        if not out_dir or not os.path.isdir(out_dir):
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_output_dir"))
            return
        out_path = os.path.join(out_dir, f"{stem}_matte.png")
        try:
            result = self.matting_canvas.get_result_rgba()
            result.save(out_path, "PNG")
            msg = self.tr("log_matting_saved", name=os.path.basename(out_path))
            self.append_log(msg)
            self._set_matting_status(msg, is_error=False)
        except Exception as e:
            self._set_matting_status(str(e), is_error=True)
            QMessageBox.warning(self, self.tr("error_title"), str(e))

    def on_batch_toggle(self, state):
        is_batch = self.check_use_batch.isChecked()
        self.spin_batch_size.setEnabled(is_batch)
        if is_batch and self.check_am.isChecked():
            self.append_log(self.tr("log_batch_warning"))

        # --- 自前ONNXモデル管理 ---
    def refresh_onnx_list(self):
        """onnx/ フォルダをスキャンしてドロップダウンを更新する"""
        current = self.combo_custom_onnx.currentText()
        self.combo_custom_onnx.blockSignals(True)
        self.combo_custom_onnx.clear()

        files = sorted(f for f in os.listdir(self.onnx_rembg_dir) if f.lower().endswith(".onnx"))
        if not files:
            self.combo_custom_onnx.addItem(self.tr("onnx_list_empty"))
        else:
            self.combo_custom_onnx.addItems(files)
            if current in files:
                self.combo_custom_onnx.setCurrentText(current)

        self.combo_custom_onnx.blockSignals(False)

    def browse_onnx_file(self):
        dialog_title = "自前ONNXモデルを選択" if self.lang == "ja" else "Select Custom ONNX Model"
        file_path, _ = QFileDialog.getOpenFileName(
            self, dialog_title, "", "ONNX Models (*.onnx)"
        )
        if not file_path:
            return

        dest_path = os.path.join(self.onnx_rembg_dir, os.path.basename(file_path))
        try:
            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                shutil.copy2(file_path, dest_path)
            self.refresh_onnx_list()
            self.combo_custom_onnx.setCurrentText(os.path.basename(dest_path))
            self.check_use_custom.setChecked(True)
            self.save_config()
        except Exception as e:
            QMessageBox.warning(self, self.tr("error_title"), self.tr("msg_onnx_copy_failed", err=e))

    def on_custom_toggle(self, state):
        is_custom = self.check_use_custom.isChecked()
        self.combo_custom_onnx.setEnabled(is_custom)
        self.btn_refresh_onnx.setEnabled(is_custom)
        self.combo_model.setEnabled(not is_custom)
        self.check_use_batch.setEnabled(is_custom)  # ← 追加
        if not is_custom:
            self.check_use_batch.setChecked(False)  # 標準モデルに戻したらバッチ推論も強制OFF

    def append_log(self, message: str):
        """ログをスクロール履歴として追記し、自動的に最下部までスクロールする"""
        self.log_view.append(message)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def cancel_batch_processing(self):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.cancel()
        self.btn_cancel.setEnabled(False)
        self.append_log(self.tr("log_cancel_requested"))

    # --- プリセット管理 ---
    def _remember_last_preset(self, name: str):
        try:
            config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            config.setdefault("rembg", {})
            config["rembg"]["last_preset"] = name
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"設定の保存エラー: {e}")

    def refresh_preset_list(self):
        current = self.combo_preset.currentText()
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()

        names = sorted(
            os.path.splitext(f)[0] for f in os.listdir(self.presets_rembg_dir) if f.lower().endswith(".json")
        )
        self.combo_preset.addItems(names)
        if current:
            self.combo_preset.setCurrentText(current)

        self.combo_preset.blockSignals(False)

    def save_preset(self):
        name = self.combo_preset.currentText().strip()
        if not name:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_preset_name"))
            return

        preset = {
            "use_custom_onnx": self.check_use_custom.isChecked(),
            "custom_onnx_filename": self.combo_custom_onnx.currentText(),
            "standard_model": self.combo_model.currentText(),
            "mask_blur": self.spin_m_blur.value(),
            "mask_threshold": self.spin_m_thresh.value(),
            "only_mask": self.check_om.isChecked(),
            "alpha_matting": self.check_am.isChecked(),
            "alpha_matting_background_threshold": self.spin_bg.value(),
            "alpha_matting_foreground_threshold": self.spin_fg.value(),
            "alpha_matting_erode_structure_size": self.spin_erode.value(),
            "spill_enabled": self.check_spill.isChecked(),
            "spill_threshold": self.spin_spill_threshold.value(),
            "spill_patch": self.spin_spill_patch.value(),
            "hole_fill_enabled": self.check_hole_fill.isChecked(),
            "hole_fill_threshold": self.spin_hole_fill_threshold.value(),
            "hole_fill_max_area": self.spin_hole_fill_max_area.value(),
            # 内部キー(bilinear/bicubic/lanczos3)で保存する(言語非依存・表示ラベルには依存しない)
            "resample_method": self.combo_resample.currentData(),
        }

        try:
            path = os.path.join(self.presets_rembg_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset, f, ensure_ascii=False, indent=4)
            self.refresh_preset_list()
            self.combo_preset.setCurrentText(name)
            self._remember_last_preset(name)
            self.append_log(self.tr("log_preset_saved", name=name))
        except Exception as e:
            QMessageBox.warning(self, self.tr("error_title"), self.tr("msg_preset_save_failed", err=e))

    def load_preset(self):
        name = self.combo_preset.currentText().strip()
        path = os.path.join(self.presets_rembg_dir, f"{name}.json")
        if not os.path.exists(path):
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_preset_not_found", name=name))
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                preset = json.load(f)

            self.check_use_custom.setChecked(preset.get("use_custom_onnx", False))
            self.refresh_onnx_list()
            onnx_name = preset.get("custom_onnx_filename", "")
            if onnx_name:
                self.combo_custom_onnx.setCurrentText(onnx_name)

            self.combo_model.setCurrentText(preset.get("standard_model", "u2net"))
            self.spin_m_blur.setValue(preset.get("mask_blur", 0))
            self.spin_m_thresh.setValue(preset.get("mask_threshold", 0))
            self.check_om.setChecked(preset.get("only_mask", False))
            self.check_am.setChecked(preset.get("alpha_matting", False))
            self.spin_bg.setValue(preset.get("alpha_matting_background_threshold", 10))
            self.spin_fg.setValue(preset.get("alpha_matting_foreground_threshold", 240))
            self.spin_erode.setValue(preset.get("alpha_matting_erode_structure_size", 10))
            self.check_spill.setChecked(preset.get("spill_enabled", False))
            self.spin_spill_threshold.setValue(preset.get("spill_threshold", 60))
            self.spin_spill_patch.setValue(preset.get("spill_patch", 10))
            self.check_hole_fill.setChecked(preset.get("hole_fill_enabled", False))
            self.spin_hole_fill_threshold.setValue(preset.get("hole_fill_threshold", 128))
            self.spin_hole_fill_max_area.setValue(preset.get("hole_fill_max_area", 500))
            resample_key = self._resolve_resample_key(preset.get("resample_method", "lanczos3"))
            idx = self.combo_resample.findData(resample_key)
            self.combo_resample.setCurrentIndex(idx if idx >= 0 else self.combo_resample.findData("lanczos3"))

            self._remember_last_preset(name)
            self.append_log(self.tr("log_preset_loaded", name=name))
        except Exception as e:
            QMessageBox.warning(self, self.tr("error_title"), self.tr("msg_preset_load_failed", err=e))

    @staticmethod
    def _resolve_resample_key(value: str) -> str:
        """新形式(内部キー)・旧形式(日本語ラベル)どちらのプリセットも読み込めるようにする。"""
        if value in RESAMPLE_KEYS:
            return value
        legacy_map = {"バイリニア": "bilinear", "バイキュービック": "bicubic", "ランチョス3": "lanczos3"}
        return legacy_map.get(value, "lanczos3")

    # ---------------- 処理ロジック ----------------
    
    # --- 新規追加: 設定の読み込み ---
    def load_config(self):
        if not os.path.exists(self.config_file):
            return
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # --- 背景除去タブ ---
            rembg_cfg = config.get("rembg", {})
            output_rembg = rembg_cfg.get("output_dir", "")
            if os.path.isdir(output_rembg):
                self.edit_output_rembg.setText(output_rembg)

            self.check_use_custom.setChecked(rembg_cfg.get("use_custom_onnx", False))
            last_onnx = rembg_cfg.get("custom_onnx_filename", "")
            if last_onnx:
                self.refresh_onnx_list()
                self.combo_custom_onnx.setCurrentText(last_onnx)

            last_preset = rembg_cfg.get("last_preset", "").strip()
            if last_preset:
                self.refresh_preset_list()
                preset_path = os.path.join(self.presets_rembg_dir, f"{last_preset}.json")
                if os.path.exists(preset_path):
                    self.combo_preset.setCurrentText(last_preset)
                    self.load_preset()

            # --- アップスケールタブ(将来用: 現状は出力先のみ) ---
            upscale_cfg = config.get("upscale", {})
            output_upscale = upscale_cfg.get("output_dir", "")
            if os.path.isdir(output_upscale):
                self.edit_output_upscale.setText(output_upscale)

            # --- マットインペイントタブ(将来用: 現状は出力先のみ) ---
            matting_cfg = config.get("matting", {})
            output_matting = matting_cfg.get("output_dir", "")
            if os.path.isdir(output_matting):
                self.edit_output_matting.setText(output_matting)

        except Exception as e:
            print(f"設定の読み込みエラー: {e}")

    def save_config(self):
        try:
            config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

            config.setdefault("rembg", {})
            config["rembg"]["output_dir"] = self.edit_output_rembg.text().strip()
            config["rembg"]["use_custom_onnx"] = self.check_use_custom.isChecked()
            config["rembg"]["custom_onnx_filename"] = self.combo_custom_onnx.currentText()

            config.setdefault("upscale", {})
            config["upscale"]["output_dir"] = self.edit_output_upscale.text().strip()

            config.setdefault("matting", {})
            config["matting"]["output_dir"] = self.edit_output_matting.text().strip()

            config["language"] = self.lang

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"設定の保存エラー: {e}")


    # 選択された画像のリアルタイムプレビュー表示
    def _show_preview(self, label: QLabel, file_path: str):
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            label.setText(self.tr("preview_unavailable"))
            return
        scaled = pixmap.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)

    def update_image_preview(self, current, previous=None):
        if current is None:
            self.label_preview_before.setText(self.tr("preview_before_empty"))
            self.label_preview_before.setPixmap(QPixmap())
            self.label_preview_after.setText(self.tr("preview_after_empty"))
            self.label_preview_after.setPixmap(QPixmap())
            return

        file_path = current.text()
        self._show_preview(self.label_preview_before, file_path)

        output_dir = self.edit_output_rembg.text().strip()
        pure_name = Path(file_path).stem
        found = None
        if output_dir:
            for suffix in ("_rembg.png", "_mask.png"):
                candidate = os.path.join(output_dir, f"{pure_name}{suffix}")
                if os.path.exists(candidate):
                    found = candidate
                    break

        if found:
            self._show_preview(self.label_preview_after, found)
        else:
            self.label_preview_after.setText(self.tr("preview_after_empty"))
            self.label_preview_after.setPixmap(QPixmap())

    # 標準モデルのコンボボックス変更時に説明文を更新
    def update_model_description(self, text):
        entry = MODEL_DESCRIPTIONS.get(text, {})
        self.model_desc.setText(entry.get(self.lang, entry.get("ja", "")))

    # --- resampleコンボの表示ラベルだけを再翻訳する(内部キー/選択状態は維持) ---
    def _retranslate_resample_combo(self, initial: bool = False):
        current_key = self.combo_resample.currentData() if not initial else self.combo_resample.currentData()
        self.combo_resample.blockSignals(True)
        for i, key in enumerate(RESAMPLE_KEYS):
            self.combo_resample.setItemText(i, self.tr(f"resample_{key}"))
        self.combo_resample.setToolTip(
            ("モデルへの入力リサイズ・マスクの拡大縮小に使用する補間方式。\n"
             "バイリニア: 高速だがやや粗い\nバイキュービック: バランス型\nランチョス3: 最も高精細だが処理はやや重い")
            if self.lang == "ja" else
            ("Interpolation method used to resize the model input / rescale the mask.\n"
             "Bilinear: fast but coarser\nBicubic: balanced\nLanczos-3: highest quality but somewhat heavier")
        )
        if current_key is not None:
            idx = self.combo_resample.findData(current_key)
            if idx >= 0:
                self.combo_resample.setCurrentIndex(idx)
        self.combo_resample.blockSignals(False)

    # 画像ファイルをリストに追加するダイアログ
    def add_files_dialog(self):
        dialog_title = "画像ファイルを追加" if self.lang == "ja" else "Add Image Files"
        files, _ = QFileDialog.getOpenFileNames(
            self, dialog_title, "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        for file in files:
            items = [self.file_list.item(i).text() for i in range(self.file_list.count())]
            if file not in items:
                self.file_list.addItem(file)

    # 選択した画像ファイルをリストから削除
    def remove_selected_items(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
            
    # バッチ処理（一括処理）の開始
    def start_batch_processing(self):
        file_count = self.file_list.count()
        if file_count == 0:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_files"))
            return

        output_dir = self.edit_output_rembg.text().strip()
        if not output_dir:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_output_dir"))
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_output_dir_not_found"))
            return

        # --- 変更: 自前ONNXのフルパスをここで解決 ---
        use_custom_onnx = self.check_use_custom.isChecked()
        custom_onnx_path = ""
        if use_custom_onnx:
            selected_file = self.combo_custom_onnx.currentText()
            if not selected_file or not selected_file.lower().endswith(".onnx"):
                QMessageBox.warning(self, self.tr("warning_title"), self.tr("msg_no_custom_onnx_selected"))
                return
            custom_onnx_path = os.path.join(self.onnx_rembg_dir, selected_file)

        options = {
            "only_mask": self.check_om.isChecked(),
            "alpha_matting": self.check_am.isChecked(),
            "alpha_matting_background_threshold": self.spin_bg.value(),
            "alpha_matting_foreground_threshold": self.spin_fg.value(),
            "alpha_matting_erode_structure_size": self.spin_erode.value(),
        }

        selected_model = self.combo_model.currentText()
        mask_blur = self.spin_m_blur.value()
        mask_threshold = self.spin_m_thresh.value()

        # --- スピル除去パラメータ ---
        spill_enabled = self.check_spill.isChecked()
        spill_threshold = self.spin_spill_threshold.value()
        spill_patch = self.spin_spill_patch.value()

        # --- 穴埋め処理パラメータ ---
        hole_fill_enabled = self.check_hole_fill.isChecked()
        hole_fill_threshold = self.spin_hole_fill_threshold.value()
        hole_fill_max_area = self.spin_hole_fill_max_area.value()

        # --- バッチ推論パラメータ ---
        use_batch_inference = self.check_use_batch.isChecked()
        batch_size = self.spin_batch_size.value() if use_batch_inference else 1

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_view.clear()
        file_paths = [self.file_list.item(i).text() for i in range(file_count)]

        # --- リサンプリング (内部キー -> PIL定数) ---
        resample_key = self.combo_resample.currentData()
        resample_method = RESAMPLE_PIL[resample_key]

        self.thread = BatchProcessThread(
            file_paths, output_dir, options, selected_model, use_custom_onnx,
            custom_onnx_path, mask_blur, mask_threshold,
            spill_enabled, spill_threshold, spill_patch,
            hole_fill_enabled, hole_fill_threshold, hole_fill_max_area,
            resample_method,  # ← 追加
            batch_size=batch_size,
            lang=self.lang,  # --- 追加: ログもUIと同じ言語で出力する ---
        )
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.on_processing_finished)
        self.thread.start()

    def on_processing_finished(self, success_count):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.append_log(log_t(self.lang, "batch_complete", count=success_count))

        current_item = self.file_list.currentItem()
        if current_item:
            self.update_image_preview(current_item)  # ← 追加：処理後画像を反映

    # def on_tab_changed(self, index: int):
    #     tab_text = self.tab_widget.tabText(index)
    #     total_width = self.main_splitter.width()
    #     if tab_text == "マットインペイント":
    #         # 左を狭く、右(作業スペース)を大きく
    #         self.main_splitter.setSizes([int(total_width * 0.22), int(total_width * 0.78)])
    #     else:
    #         # 通常比率に戻す
    #         self.main_splitter.setSizes([int(total_width * 0.55), int(total_width * 0.45)])

    # def _build_matting_tab(self) -> QWidget:
    #     tab = QWidget()
    #     layout = QHBoxLayout(tab)

    #     # --- 左: 出力元切り替え + ファイル一覧(タブ内専用) ---
    #     source_panel = QVBoxLayout()

    #     self.combo_matting_source = QComboBox()
    #     self.combo_matting_source.addItems(["背景除去の出力 (output/rembg/)", "アップスケールの出力 (output/upscale/)"])
    #     self.combo_matting_source.currentIndexChanged.connect(self.refresh_matting_file_list)
    #     source_panel.addWidget(QLabel("編集対象の出力元:"))
    #     source_panel.addWidget(self.combo_matting_source)

    #     self.matting_file_list = QListWidget()
    #     self.matting_file_list.currentItemChanged.connect(self.load_image_to_matting_canvas)
    #     source_panel.addWidget(self.matting_file_list, stretch=1)

    #     btn_refresh = QPushButton("一覧を更新")
    #     btn_refresh.clicked.connect(self.refresh_matting_file_list)
    #     source_panel.addWidget(btn_refresh)

    #     source_container = QWidget()
    #     source_container.setLayout(source_panel)
    #     source_container.setMaximumWidth(220)  # 細めに固定してキャンバス側を広く保つ
    #     layout.addWidget(source_container)

    #     # --- 右: 編集キャンバス(大きめ) ---
    #     self.matting_canvas = MattingCanvas()  # 今後実装するQPainterベースの編集ウィジェット
    #     layout.addWidget(self.matting_canvas, stretch=1)

    #     return tab

    # def refresh_matting_file_list(self):
    #     source_index = self.combo_matting_source.currentIndex()
    #     target_dir = self.edit_output_rembg.text() if source_index == 0 else self.edit_output_upscale.text()

    #     self.matting_file_list.clear()
    #     if not os.path.isdir(target_dir):
    #         return

    #     valid_ext = (".png", ".jpg", ".jpeg", ".webp")
    #     files = sorted(f for f in os.listdir(target_dir) if f.lower().endswith(valid_ext))
    #     for f in files:
    #         self.matting_file_list.addItem(os.path.join(target_dir, f))

    def _build_output_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- 表示言語切り替え ---
        lang_group = QGroupBox()
        self._reg(lang_group, "language_group", kind="title")
        lang_form = QFormLayout()
        lang_form.setContentsMargins(5, 5, 5, 5)

        self.combo_language = QComboBox()
        self.combo_language.addItem("日本語", "ja")
        self.combo_language.addItem("English", "en")
        self.combo_language.setCurrentIndex(self.combo_language.findData(self.lang))
        self.combo_language.currentIndexChanged.connect(self.on_language_changed)
        lang_form.addRow(self._reg(QLabel(), "language_label"), self.combo_language)
        lang_group.setLayout(lang_form)
        layout.addWidget(lang_group)

        # 出力先グループ(タブ名変更時と同じキーを使い回すことで表記を統一する)
        self.output_dir_groups = []  # [(QGroupBox, label_key), ...] retranslate_ui()で見出しを更新

        self.edit_output_rembg, group_rembg = self._make_output_dir_group(
            "output_group_rembg", self.default_output_rembg
        )
        self.edit_output_upscale, group_upscale = self._make_output_dir_group(
            "output_group_upscale", self.default_output_upscale
        )
        self.edit_output_matting, group_matting = self._make_output_dir_group(
            "output_group_matting", self.default_output_matting
        )

        layout.addWidget(group_rembg)
        layout.addWidget(group_upscale)
        layout.addWidget(group_matting)
        layout.addStretch(1)
        return tab

    def _make_output_dir_group(self, label_key: str, default_dir: str):
        group = QGroupBox()
        self.output_dir_groups.append((group, label_key))
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)

        edit = QLineEdit()
        edit.setText(default_dir)
        btn = QPushButton()
        self._reg(btn, "btn_browse")
        btn.clicked.connect(lambda: self._browse_output_dir_for(edit))

        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        form.addRow(self._reg(QLabel(), "output_save_to_label"), row)
        group.setLayout(form)
        self._retranslate_output_dir_groups()
        return edit, group

    def _retranslate_output_dir_groups(self):
        """出力先グループの見出しは『機能名 + の出力先』の組み合わせなので専用に再構築する。"""
        for group, label_key in getattr(self, "output_dir_groups", []):
            group.setTitle(self.tr(label_key) + self.tr("output_group_suffix"))

    def _browse_output_dir_for(self, line_edit: QLineEdit):
        current = line_edit.text().strip()
        dir_path = QFileDialog.getExistingDirectory(self, self.tr("browse_dir_dialog_title"), current)
        if dir_path:
            line_edit.setText(dir_path)
            self.save_config()

class PlaceholderTab(QWidget):
    """未実装機能のプレースホルダー。feature_key は tr()用の翻訳キー(例: 'tab_upscale')。"""

    def __init__(self, feature_key: str, lang: str = "ja"):
        super().__init__()
        self.feature_key = feature_key
        layout = QVBoxLayout(self)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888;")
        layout.addStretch(1)
        layout.addWidget(self.label)
        layout.addStretch(1)
        self.retranslate(lang)

    def retranslate(self, lang: str):
        feature_name = tr(lang, self.feature_key)
        self.label.setText(tr(lang, "placeholder_text", name=feature_name))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 修正: QApplication構築後にsetStyleで確実に適用
    window = RembgGuiApp()
    window.show()
    sys.exit(app.exec())
