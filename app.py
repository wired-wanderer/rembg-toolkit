import sys
import os
import io
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QFileDialog, QGroupBox, QFormLayout, 
    QSpinBox, QCheckBox, QLabel, QProgressBar, QMessageBox, QComboBox, QLineEdit,
    QTextEdit, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from rembg import remove, new_session
from rembg.sessions.base import BaseSession
from PIL import Image, ImageFilter
import numpy as np
import shutil

# 自前ONNXファイルを強制的に読み込ませるためのカスタムセッションクラス
# 内部モジュールから直接 BiRefNetSession を強制インポートします

class CustomOnnxSession(BaseSession):
    """
    自前ONNXファイル(BiRefNet系)を読み込むためのカスタムセッション。

    注意: BaseSession.__init__() をそのまま呼ぶと、内部の download_models() が
    未登録モデル名で呼ばれて失敗する可能性がある。また inner_session を
    「メソッド」のままにすると BaseSession 側の期待する「InferenceSessionインスタンス」
    と衝突するため、ここでは super().__init__() に頼らず自前で完結させる。
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = "custom-birefnet"

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self._build_providers()

        # inner_session は「メソッド」ではなく実インスタンスとして持つ
        self.inner_session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
        )

        active_ep = self.inner_session.get_providers()
        print(f"[CustomOnnxSession] Active Execution Provider: {active_ep}")
        if "CUDAExecutionProvider" not in active_ep:
            print("[警告] CUDA EPが有効化されていません。CPUにフォールバックしています。")

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
        img_resized = img.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        img_np = img_np.transpose((2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0).astype(np.float32)

        input_name = self.inner_session.get_inputs()[0].name
        outputs = self.inner_session.run(None, {input_name: img_np})

        pred = outputs[0][0][0]
        # 修正: モデル自体が既に0.0〜1.0の確率値を出力しているため、シグモイドは再適用しない
        mask_np = (np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8)

        mask_img = Image.fromarray(mask_np, mode="L")
        mask_img = mask_img.resize((w, h), Image.Resampling.BILINEAR)
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
            img_resized = img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            arr = arr.transpose((2, 0, 1))
            batch_np.append(arr)

        batch_tensor = np.stack(batch_np, axis=0).astype(np.float32)  # (N, 3, 1024, 1024)

        input_name = self.inner_session.get_inputs()[0].name
        outputs = self.inner_session.run(None, {input_name: batch_tensor})[0]  # (N, 1, 1024, 1024)

        mask_imgs = []
        for i, (w, h) in enumerate(sizes):
            pred = outputs[i][0]
            mask_np = (np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np, mode="L").resize((w, h), Image.Resampling.LANCZOS)
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
                 spill_enabled, spill_threshold, spill_patch, batch_size=1):
        super().__init__()
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
        self.batch_size = batch_size  # ← 追加
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

    # --- 共通後処理ヘルパー ---
    def _postprocess_alpha(self, orig_img: Image.Image, alpha: Image.Image) -> Image.Image:
        if self.spill_enabled:
            alpha = self._suppress_color_spill(
                orig_img, alpha, threshold=self.spill_threshold, patch=self.spill_patch
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
                self.log_signal.emit(f"⏹ キャンセルされました。（{index}/{total} 件処理済み）")
                break
            try:
                self.log_signal.emit(f"処理中 ({index+1}/{total}): {os.path.basename(file_path)} ...")
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
                self.log_signal.emit(f"【成功】保存先: {os.path.basename(out_path)}")
            except Image.DecompressionBombError:
                msg = f"【エラー】{os.path.basename(file_path)}: 画像サイズが大きすぎます"
                self.log_signal.emit(msg); print(msg)
            except Exception as e:
                msg = f"【エラー】{os.path.basename(file_path)}: {str(e)}"
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
                self.log_signal.emit(f"⏹ キャンセルされました。（{processed}/{total} 件処理済み）")
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
                    msg = f"【エラー】{os.path.basename(file_path)}: 読み込み失敗 ({str(e)})"
                    self.log_signal.emit(msg); print(msg)
                    self.progress_signal.emit(int((processed / total) * 100))

            if not orig_imgs:
                continue

            mask_imgs = None
            if batch_supported:
                self.log_signal.emit(f"バッチ {batch_num}/{total_batches} を推論中 ({len(orig_imgs)}枚)...")
                try:
                    mask_imgs = session.predict_batch(orig_imgs)
                except Exception as e:
                    # --- 変更: このモデルはバッチ非対応と判断し、以降は最初から個別推論に切り替える ---
                    batch_supported = False
                    msg = (
                        f"【情報】このモデルはバッチ推論(複数枚同時投入)に対応していません"
                        f"（{str(e).splitlines()[0]}）。以降は1枚ずつ処理します。"
                    )
                    self.log_signal.emit(msg)
                    print(msg)

            if mask_imgs is None:
                self.log_signal.emit(f"バッチ {batch_num}/{total_batches} を1枚ずつ推論中 ({len(orig_imgs)}枚)...")
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
                    self.log_signal.emit(f"【エラー】{os.path.basename(file_path)}: マスク生成に失敗しました")
                    self.progress_signal.emit(int((processed / total) * 100))
                    continue
                try:
                    alpha = self._postprocess_alpha(orig_img, mask_img)
                    out_path = self._save_result(file_path, orig_img, alpha)
                    success_count += 1
                    self.log_signal.emit(f"【成功】保存先: {os.path.basename(out_path)}")
                except Exception as e:
                    msg = f"【エラー】{os.path.basename(file_path)}: {str(e)}"
                    self.log_signal.emit(msg); print(msg); print(traceback.format_exc())

                self.progress_signal.emit(int((processed / total) * 100))

        return success_count

    def run(self):
        total = len(self.file_paths)
        try:
            if self.use_custom_onnx and self.custom_onnx_path and os.path.exists(self.custom_onnx_path):
                self.log_signal.emit(f"カスタムONNXモデル（{os.path.basename(self.custom_onnx_path)}）を読み込み中...")
                session = CustomOnnxSession(self.custom_onnx_path)
            else:
                self.log_signal.emit(f"標準AIモデル（{self.model_name}）を読み込み中...")
                session = new_session(self.model_name)
        except Exception as e:
            msg = f"【エラー】モデルの読み込みに失敗しました: {str(e)}"
            self.log_signal.emit(msg); print(msg); print(traceback.format_exc())
            self.finished_signal.emit(0)
            return

        use_batch = self.use_custom_onnx and isinstance(session, CustomOnnxSession) and self.batch_size > 1

        if use_batch:
            self.log_signal.emit(
                f"バッチ推論モード (バッチサイズ={self.batch_size}) で処理します。"
                f"※rembgのアルファマッティング後処理はこのモードではスキップされます。"
            )
            success_count = self._run_batched(session, total)
        else:
            success_count = self._run_sequential(session, total)

        del session
        import gc
        gc.collect()

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
# メインウィンドウクラス
class RembgGuiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rembg-toolkit 高機能一括背景透過ツール")
        self.resize(1300, 850)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(base_dir, "config.json")

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

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- 左側: 共通パネル(ファイルリスト・プレビュー・ログ) ---
        left_layout = self._build_shared_left_panel()
        main_layout.addLayout(left_layout, stretch=4)

        # --- 右側: タブウィジェット ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, stretch=3)

        self.tab_widget.addTab(self._build_rembg_tab(), "背景除去")
        self.tab_widget.addTab(PlaceholderTab("アップスケール"), "アップスケール")
        self.tab_widget.addTab(PlaceholderTab("マットインペイント"), "マットインペイント")
        self.tab_widget.addTab(self._build_output_settings_tab(), "保存先設定")

    def _build_shared_left_panel(self) -> QVBoxLayout:
        # ---------------- 左側: ファイルリスト ----------------
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("処理する画像ファイル (ここにドラッグ＆ドロップ)"))

        self.file_list = DropListWidget()
        left_layout.addWidget(self.file_list, stretch=2) # サイズ割合

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("ファイルを追加")
        self.btn_add.clicked.connect(self.add_files_dialog)
        self.btn_remove = QPushButton("選択削除")
        self.btn_remove.clicked.connect(self.remove_selected_items)
        self.btn_clear = QPushButton("リストクリア")
        self.btn_clear.clicked.connect(self.file_list.clear)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_clear)
        left_layout.addLayout(btn_layout)

        # --- 変更: プレビューグループをここ(中段)に移動 ---
        preview_group = QGroupBox("プレビュー比較 (処理前 / 処理後)")
        preview_layout = QHBoxLayout()

        self.label_preview_before = QLabel("処理前\n(ファイル未選択)")
        self.label_preview_before.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview_before.setMinimumSize(150, 150)  # 左パネル幅に合わせて縮小
        self.label_preview_before.setStyleSheet("background-color: #2b2b2b; color: #ccc; border: 1px solid #555;")

        self.label_preview_after = QLabel("処理後\n(未処理)")
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
        left_layout.addWidget(QLabel("処理ログ:"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("ステータス: 待機中")
        left_layout.addWidget(self.log_view, stretch=1)  # サイズ割合

        self.file_list.currentItemChanged.connect(self.update_image_preview)
        return left_layout

    def _build_rembg_tab(self) -> QWidget:
        """既存の「自前ONNXモデル」「マスクエッジ調整」「スピル除去」「パフォーマンス」
        「プリセット」「実行/キャンセルボタン」の各グループボックスをまとめて1つのタブにする"""
        tab = QWidget()
        right_layout = QVBoxLayout(tab)

        # ---------------- 右側: 設定オプション（省スペース設計） ----------------
        right_layout.setSpacing(5)  # 各ウィジェット間の隙間を詰める

        # --- プリセットグループ ---
        preset_group = QGroupBox("設定プリセット")
        preset_form = QFormLayout()
        preset_form.setContentsMargins(5, 5, 5, 5)

        self.combo_preset = QComboBox()
        self.combo_preset.setEditable(True)
        self.combo_preset.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        preset_form.addRow(QLabel("プリセット名:"), self.combo_preset)

        preset_btn_layout = QHBoxLayout()
        self.btn_save_preset = QPushButton("保存")
        self.btn_save_preset.clicked.connect(self.save_preset)
        self.btn_load_preset = QPushButton("読み込み")
        self.btn_load_preset.clicked.connect(self.load_preset)
        self.btn_refresh_preset = QPushButton("更新")
        self.btn_refresh_preset.clicked.connect(self.refresh_preset_list)

        preset_btn_layout.addWidget(self.btn_save_preset)
        preset_btn_layout.addWidget(self.btn_load_preset)
        preset_btn_layout.addWidget(self.btn_refresh_preset)
        preset_form.addRow(preset_btn_layout)

        preset_group.setLayout(preset_form)
        right_layout.addWidget(preset_group)

        # --- 1. 自前ONNXモデル選択グループ ---
        custom_group = QGroupBox("自前ONNXモデルの使用 (ToonOutなど)")
        custom_form = QFormLayout()
        custom_form.setContentsMargins(5, 5, 5, 5)

        # --- パフォーマンス設定 ---

        perf_group = QGroupBox("パフォーマンス設定")
        perf_form = QFormLayout()
        perf_form.setContentsMargins(5, 5, 5, 5)

        self.check_use_batch = QCheckBox("バッチ推論を使用する（自前ONNXモデル専用）")
        self.check_use_batch.setChecked(False)
        self.check_use_batch.setToolTip(
            "複数枚をまとめてGPUに投げることで高速化しますが、\n"
            "rembgのアルファマッティング後処理は適用されなくなります。"
        )
        self.check_use_batch.stateChanged.connect(self.on_batch_toggle)
        perf_form.addRow(self.check_use_batch)

        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(2, 32)
        self.spin_batch_size.setValue(4)
        self.spin_batch_size.setEnabled(False)  # 初期はチェックボックスOFFなので無効
        self.spin_batch_size.setToolTip("まとめてGPUに投げる画像数。大きいほど高速だがVRAM消費も増える。")
        perf_form.addRow(QLabel("バッチサイズ:"), self.spin_batch_size)

        perf_group.setLayout(perf_form)
        right_layout.addWidget(perf_group)

        # 自前/標準切り替えチェックボックス
        self.check_use_custom = QCheckBox("自前ONNXモデルを使用する")
        self.check_use_custom.setChecked(False)
        self.check_use_custom.stateChanged.connect(self.on_custom_toggle)
        custom_form.addRow(self.check_use_custom)

        # ドロップダウン + 更新ボタン
        self.combo_custom_onnx = QComboBox()
        self.btn_refresh_onnx = QPushButton("更新")
        self.btn_refresh_onnx.setToolTip(f"{self.onnx_rembg_dir} 内の.onnxファイルを再スキャンします")
        self.btn_refresh_onnx.clicked.connect(self.refresh_onnx_list)

        onnx_combo_layout = QHBoxLayout()
        onnx_combo_layout.addWidget(self.combo_custom_onnx)
        onnx_combo_layout.addWidget(self.btn_refresh_onnx)
        custom_form.addRow(QLabel("モデル選択:"), onnx_combo_layout)

        # 外部ファイルを選択 → onnx/ フォルダへ自動コピーして登録
        self.btn_browse_onnx = QPushButton("外部からファイルを追加...")
        self.btn_browse_onnx.setToolTip(f"選択したファイルを {self.onnx_rembg_dir} にコピーして一覧に追加します")
        self.btn_browse_onnx.clicked.connect(self.browse_onnx_file)
        custom_form.addRow(self.btn_browse_onnx)

        custom_group.setLayout(custom_form)
        right_layout.addWidget(custom_group)
        
        # --- 2. 標準AIモデル選択グループ ---
        model_group = QGroupBox("標準AIモデルの選択 (自前ONNXが空のとき適用)")
        model_form = QFormLayout()
        model_form.setContentsMargins(5, 5, 5, 5)
        
        self.combo_model = QComboBox()
        models = [
            "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta",
            "isnet-general-use", "isnet-anime", "sam",
            "birefnet-general", "birefnet-general-lite", "birefnet-portrait",
            "birefnet-dis", "birefnet-hrsod", "birefnet-cod", "birefnet-massive",
            "bria-rmbg [非商用限定]"
        ]
        self.combo_model.addItems(models)
        self.combo_model.setCurrentText("u2net")
        
        self.model_desc = QLabel("一般的な用途向けの事前学習済みモデル（デフォルト）")
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet("color: #555555; font-size: 11px;")
        self.combo_model.currentTextChanged.connect(self.update_model_description)

        model_form.addRow(QLabel("モデル:"), self.combo_model)
        model_form.addRow(self.model_desc)
        model_group.setLayout(model_form)
        right_layout.addWidget(model_group)

        # --- 3. マスクのダイレクト調整グループ ---
        mask_adjust_group = QGroupBox("マスクエッジの直接調整")
        mask_form = QFormLayout()
        mask_form.setContentsMargins(5, 5, 5, 5)

        self.spin_m_blur = QSpinBox()
        self.spin_m_blur.setRange(0, 50)
        self.spin_m_blur.setValue(0)
        mask_form.addRow(QLabel("マスクブラー (Blur):"), self.spin_m_blur)

        self.spin_m_thresh = QSpinBox()
        self.spin_m_thresh.setRange(-255, 255)  # ← マイナス値を受け付ける
        self.spin_m_thresh.setValue(0)          # 初期値は 0
        mask_form.addRow(QLabel("マスクオフセット (Offset):"), self.spin_m_thresh)

        mask_adjust_group.setLayout(mask_form)
        right_layout.addWidget(mask_adjust_group)

        # --- 3.5 色スピル除去グループ ---
        spill_group = QGroupBox("背景色スピル除去 (グリーンバック等の色にじみ対策)")
        spill_form = QFormLayout()
        spill_form.setContentsMargins(5, 5, 5, 5)

        self.check_spill = QCheckBox("有効にする")
        self.check_spill.setChecked(True)
        spill_form.addRow(QLabel("スピル除去:"), self.check_spill)

        self.spin_spill_threshold = QSpinBox()
        self.spin_spill_threshold.setRange(0, 255)
        self.spin_spill_threshold.setValue(60)
        self.spin_spill_threshold.setToolTip("大きいほど背景色に近い色まで広く透明化します")
        spill_form.addRow(QLabel("色差しきい値:"), self.spin_spill_threshold)

        self.spin_spill_patch = QSpinBox()
        self.spin_spill_patch.setRange(1, 100)
        self.spin_spill_patch.setValue(10)
        self.spin_spill_patch.setToolTip("背景色を推定するために画像四隅から取るサンプル領域の大きさ(px)")
        spill_form.addRow(QLabel("背景色サンプル幅(px):"), self.spin_spill_patch)

        spill_group.setLayout(spill_form)
        right_layout.addWidget(spill_group)

        # --- 4. 詳細設定グループ（アルファマッティング） ---
        option_group = QGroupBox("アルファマッティング（AI自動補正）")
        form_layout = QFormLayout()
        form_layout.setContentsMargins(5, 5, 5, 5)

        self.check_om = QCheckBox("有効にする (-om)")
        self.check_om.setChecked(False)
        form_layout.addRow(QLabel("マスクのみ出力:"), self.check_om)

        self.check_am = QCheckBox("有効にする")
        self.check_am.setChecked(True)
        form_layout.addRow(QLabel("アルファ補正:"), self.check_am)

        self.spin_bg = QSpinBox()
        self.spin_bg.setRange(0, 255)
        self.spin_bg.setValue(10)
        form_layout.addRow(QLabel("背景閾値 (bg):"), self.spin_bg)

        self.spin_fg = QSpinBox()
        self.spin_fg.setRange(0, 255)
        self.spin_fg.setValue(240)
        form_layout.addRow(QLabel("前景閾値 (fg):"), self.spin_fg)

        self.spin_erode = QSpinBox()
        self.spin_erode.setRange(0, 100)
        self.spin_erode.setValue(10)
        form_layout.addRow(QLabel("侵食サイズ (erode):"), self.spin_erode)

        option_group.setLayout(form_layout)
        right_layout.addWidget(option_group)

        # 実行ボタン + キャンセルボタン
        right_layout.addSpacing(5)
        btn_run_layout = QHBoxLayout()

        self.btn_run = QPushButton("一括処理（バッチ）を開始")
        self.btn_run.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #2b8a3e; color: white; padding: 8px;")
        self.btn_run.clicked.connect(self.start_batch_processing)

        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #c92a2a; color: white; padding: 8px;")
        self.btn_cancel.clicked.connect(self.cancel_batch_processing)

        btn_run_layout.addWidget(self.btn_run)
        btn_run_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(btn_run_layout)

        right_layout.addStretch(1)
        return tab

    def on_batch_toggle(self, state):
        is_batch = self.check_use_batch.isChecked()
        self.spin_batch_size.setEnabled(is_batch)
        if is_batch and self.check_am.isChecked():
            self.append_log(
                "⚠ バッチ推論を有効にすると、rembgのアルファマッティング後処理は適用されません。"
            )

        # --- 自前ONNXモデル管理 ---
    def refresh_onnx_list(self):
        """onnx/ フォルダをスキャンしてドロップダウンを更新する"""
        current = self.combo_custom_onnx.currentText()
        self.combo_custom_onnx.blockSignals(True)
        self.combo_custom_onnx.clear()

        files = sorted(f for f in os.listdir(self.onnx_rembg_dir) if f.lower().endswith(".onnx"))
        if not files:
            self.combo_custom_onnx.addItem("(onnx/ にファイルがありません)")
        else:
            self.combo_custom_onnx.addItems(files)
            if current in files:
                self.combo_custom_onnx.setCurrentText(current)

        self.combo_custom_onnx.blockSignals(False)

    def browse_onnx_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "自前ONNXモデルを選択", "", "ONNX Models (*.onnx)"
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
            QMessageBox.warning(self, "エラー", f"ファイルのコピーに失敗しました: {e}")

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

    def browse_output_dir(self):
        current = self.edit_output_rembg.text().strip() or getattr(self, "last_output_dir", "")
        dir_path = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択", current)
        if dir_path:
            self.edit_output_dir.setText(dir_path)
            self.last_output_dir = dir_path
            self.save_config()

    def cancel_batch_processing(self):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.cancel()
        self.btn_cancel.setEnabled(False)
        self.append_log("⏸ キャンセル要求を送信しました。現在のファイルの処理完了後に停止します。") 

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
            QMessageBox.warning(self, "警告", "プリセット名を入力してください。")
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
        }

        try:
            path = os.path.join(self.presets_rembg_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset, f, ensure_ascii=False, indent=4)
            self.refresh_preset_list()
            self.combo_preset.setCurrentText(name)
            self._remember_last_preset(name)
            self.append_log(f"プリセット「{name}」を保存しました。")
        except Exception as e:
            QMessageBox.warning(self, "エラー", f"プリセットの保存に失敗しました: {e}")

    def load_preset(self):
        name = self.combo_preset.currentText().strip()
        path = os.path.join(self.presets_rembg_dir, f"{name}.json")
        if not os.path.exists(path):
            QMessageBox.warning(self, "警告", f"プリセット「{name}」が見つかりません。")
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
            self.check_am.setChecked(preset.get("alpha_matting", True))
            self.spin_bg.setValue(preset.get("alpha_matting_background_threshold", 10))
            self.spin_fg.setValue(preset.get("alpha_matting_foreground_threshold", 240))
            self.spin_erode.setValue(preset.get("alpha_matting_erode_structure_size", 10))
            self.check_spill.setChecked(preset.get("spill_enabled", True))
            self.spin_spill_threshold.setValue(preset.get("spill_threshold", 60))
            self.spin_spill_patch.setValue(preset.get("spill_patch", 10))

            self._remember_last_preset(name)
            self.append_log(f"プリセット「{name}」を読み込みました。")
        except Exception as e:
            QMessageBox.warning(self, "エラー", f"プリセットの読み込みに失敗しました: {e}") 

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

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"設定の保存エラー: {e}")


    # 選択された画像のリアルタイムプレビュー表示
    def _show_preview(self, label: QLabel, file_path: str):
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            label.setText("(表示できません)")
            return
        scaled = pixmap.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)

    def update_image_preview(self, current, previous=None):
        if current is None:
            self.label_preview_before.setText("処理前\n(ファイル未選択)")
            self.label_preview_before.setPixmap(QPixmap())
            self.label_preview_after.setText("処理後\n(未処理)")
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
            self.label_preview_after.setText("処理後\n(未処理)")
            self.label_preview_after.setPixmap(QPixmap())

    # 標準モデルのコンボボックス変更時に説明文を更新
    def update_model_description(self, text):
        descriptions = {
            "u2net": "一般的な用途向けの事前学習済みモデル（万能・標準）",
            "u2netp": "u2netモデルの軽量版。速度重視、省メモリ環境向け",
            "u2net_human_seg": "人間のセグメンテーション用の事前学習済みモデル（髪や体に強い）",
            "u2net_cloth_seg": "人間の肖像から衣服（上半身/下半身/全身）の解析・分離を行うモデル",
            "silueta": "u2netと同精度で、サイズを43MBに縮小した軽量化モデル",
            "isnet-general-use": "一般的な用途向けの新しい高精度な事前学習済みモデル",
            "isnet-anime": "アニメキャラクター・2Dイラストの高精度セグメンテーション特化",
            "sam": "Meta開発。あらゆる用途に対応した汎用セグメンテーションモデル",
            "birefnet-general": "近年追加された非常に強力で高精度な一般用途向けモデル",
            "birefnet-general-lite": "birefnet-generalの軽量化・高速化モデル",
            "birefnet-portrait": "人物の証明写真や顔写真（ポートレート）に特化したモデル",
            "birefnet-dis": "高精度な二分画像セグメンテーション（DIS）用モデル",
            "birefnet-hrsod": "高解像度な顕著物体検出（HRSOD）用モデル",
            "birefnet-cod": "背景に溶け込んだ「隠された物体（COD）」の検出用モデル",
            "birefnet-massive": "大規模データセットを用いて訓練された最高峰の大型モデル",
            "bria-rmbg": "BRIA AIによる、最先端の背景除去モデル"
        }
        self.model_desc.setText(descriptions.get(text, ""))

    # 画像ファイルをリストに追加するダイアログ
    def add_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "画像ファイルを追加", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
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
            QMessageBox.warning(self, "警告", "処理するファイルがリストに登録されていません。")
            return

        output_dir = self.edit_output_rembg.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "警告", "出力先フォルダが設定されていません。右側の「出力先フォルダ」から設定してください。")
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, "警告", "指定された出力先フォルダが見つかりません。パスを確認してください。")
            return

        # --- 変更: 自前ONNXのフルパスをここで解決 ---
        use_custom_onnx = self.check_use_custom.isChecked()
        custom_onnx_path = ""
        if use_custom_onnx:
            selected_file = self.combo_custom_onnx.currentText()
            if not selected_file or not selected_file.lower().endswith(".onnx"):
                QMessageBox.warning(self, "警告", "自前ONNXモデルが選択されていません。onnx/フォルダにファイルを追加してください。")
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

        # --- 追加: スピル除去パラメータ ---
        spill_enabled = self.check_spill.isChecked()
        spill_threshold = self.spin_spill_threshold.value()
        spill_patch = self.spin_spill_patch.value()

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_view.clear()
        file_paths = [self.file_list.item(i).text() for i in range(file_count)]

        # --- 変更: チェックボックスで明示的に判定 ---
        use_batch_inference = self.check_use_batch.isChecked()
        batch_size = self.spin_batch_size.value() if use_batch_inference else 1

        self.thread = BatchProcessThread(
            file_paths, output_dir, options, selected_model, use_custom_onnx,
            custom_onnx_path, mask_blur, mask_threshold,
            spill_enabled, spill_threshold, spill_patch,
            batch_size=batch_size
        )
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.on_processing_finished)
        self.thread.start()

    def on_processing_finished(self, success_count):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.append_log(f"=== 完了: 処理に成功したファイル数 {success_count} 個 ===")

        current_item = self.file_list.currentItem()
        if current_item:
            self.update_image_preview(current_item)  # ← 追加：処理後画像を反映

    def _build_output_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.edit_output_rembg, group_rembg = self._make_output_dir_group(
            "背景除去", self.default_output_rembg
        )
        self.edit_output_upscale, group_upscale = self._make_output_dir_group(
            "アップスケール", self.default_output_upscale
        )
        self.edit_output_matting, group_matting = self._make_output_dir_group(
            "マットインペイント", self.default_output_matting
        )

        layout.addWidget(group_rembg)
        layout.addWidget(group_upscale)
        layout.addWidget(group_matting)
        layout.addStretch(1)
        return tab

    def _make_output_dir_group(self, label: str, default_dir: str):
        group = QGroupBox(f"{label} の出力先")
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)

        edit = QLineEdit()
        edit.setText(default_dir)
        btn = QPushButton("選択...")
        btn.clicked.connect(lambda: self._browse_output_dir_for(edit))

        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        form.addRow(QLabel("保存先:"), row)
        group.setLayout(form)
        return edit, group

    def _browse_output_dir_for(self, line_edit: QLineEdit):
        current = line_edit.text().strip()
        dir_path = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択", current)
        if dir_path:
            line_edit.setText(dir_path)
            self.save_config()

class PlaceholderTab(QWidget):
    def __init__(self, feature_name: str):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(f"🚧 「{feature_name}」は実装予定です 🚧")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888;")
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 修正: QApplication構築後にsetStyleで確実に適用
    window = RembgGuiApp()
    window.show()
    sys.exit(app.exec())
