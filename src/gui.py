import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from sr.tiler import SuperResolutionTiler
from app.platform_check import windows_version_notice_required, WINDOWS_10_1903_BUILD, windows_version_text
from app.version_info import APP_NAME, APP_VERSION
from app.worker import InferenceWorker
from sr.util import tiler_default, upscale_bicubic
from ui.image_compare import ComparisonWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.sr: SuperResolutionTiler | None = None
        self.input_image_path: str | None = None
        self.model_path: str | None = None
        self.output_qimg: QImage | None = None

        self.comparison = ComparisonWidget()
        self.comparison.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")

        self.btn_load_model = QPushButton("1. Load model")
        self.btn_load_image = QPushButton("2. Load image")
        self.btn_run = QPushButton("3. Run")
        self.btn_run.setEnabled(False)
        self.btn_save = QPushButton("4. Save image")
        self.btn_save.setEnabled(False)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(16, 4096)
        self.tile_spin.setSingleStep(8)
        self.tile_spin.setValue(tiler_default("tile"))

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 2048)
        self.overlap_spin.setSingleStep(4)
        self.overlap_spin.setValue(tiler_default("overlap"))

        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 16)
        self.scale_spin.setValue(tiler_default("scale"))

        self.device_id_spin = QSpinBox()
        self.device_id_spin.setRange(0, 64)
        self.device_id_spin.setValue(tiler_default("device_id"))

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        model_params_box = QGroupBox("Model parameters")
        model_params_layout = QFormLayout(model_params_box)
        model_params_layout.addRow("Tile size", self.tile_spin)
        model_params_layout.addRow("Overlap", self.overlap_spin)
        model_params_layout.addRow("Scale", self.scale_spin)
        model_params_layout.addRow("Device ID", self.device_id_spin)

        control_box = QGroupBox("Control flow")
        control_layout = QVBoxLayout(control_box)
        control_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(self.btn_load_image)
        control_layout.addWidget(self.btn_run)
        control_layout.addWidget(self.btn_save)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_label)

        left_layout.addWidget(model_params_box)
        left_layout.addWidget(control_box)
        left_layout.addStretch()

        info_label = QLabel(
            'This is a ONNX model inference demo on MSW\'s DirectML (DirectX 12), '
            'from "Activating More Pixels in Image Super-Resolution Transformer" '
            '(CVPR 2023) and "HAT: Hybrid Attention Transformer for Image Restoration" '
            '(arXiv 2023).'
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        left_layout.addWidget(info_label)

        preview_box = QGroupBox("Image preview")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.comparison, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(preview_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 880])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.btn_load_model.clicked.connect(self._on_load_model)
        self.btn_load_image.clicked.connect(self._on_load_image)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_save.clicked.connect(self._on_save)
        self.tile_spin.valueChanged.connect(self._on_settings_changed)
        self.overlap_spin.valueChanged.connect(self._on_settings_changed)
        self.scale_spin.valueChanged.connect(self._on_settings_changed)
        self.device_id_spin.valueChanged.connect(self._on_settings_changed)

        self.thread: QThread | None = None
        self.worker: InferenceWorker | None = None

    def _current_tile(self) -> int:
        return self.tile_spin.value()

    def _current_overlap(self) -> int:
        return self.overlap_spin.value()

    def _current_scale(self) -> int:
        return self.scale_spin.value()

    def _current_device_id(self) -> int:
        return self.device_id_spin.value()

    def _build_tiler(self, model_path: str) -> SuperResolutionTiler:
        return SuperResolutionTiler(
            model_path=model_path,
            tile=self._current_tile(),
            overlap=self._current_overlap(),
            scale=self._current_scale(),
            device_id=self._current_device_id(),
        )

    def _apply_settings_to_loaded_tiler(self):
        if self.sr is None:
            return
        self.sr.tile = self._current_tile()
        self.sr.overlap = self._current_overlap()
        self.sr.scale = self._current_scale()
        self.sr.device_id = self._current_device_id()

    def _on_settings_changed(self, *args):
        if self.overlap_spin.value() >= self.tile_spin.value():
            self.overlap_spin.setValue(max(0, self.tile_spin.value() - 1))
            return

        self._apply_settings_to_loaded_tiler()
        self.output_qimg = None
        self.btn_save.setEnabled(False)

        if self.input_image_path is not None:
            self._refresh_input_preview()

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX model", "", "ONNX Files (*.onnx);;All Files (*)"
        )
        if not path:
            return

        try:
            self.model_path = path
            self.sr = self._build_tiler(path)
            self.sr.load()
            self.status_bar.showMessage(f"Model loaded: {Path(path).name}")
            self.btn_run.setEnabled(self.input_image_path is not None)
            if self.input_image_path is not None:
                self._refresh_input_preview()
        except Exception as e:
            self.status_bar.showMessage(f"Error loading model: {e}")

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )
        if not path:
            return

        self.input_image_path = path
        self.output_qimg = None
        self.comparison.set_output_image(None)
        self.status_bar.showMessage(f"Input image: {Path(path).name}")
        self._refresh_input_preview()
        self.btn_run.setEnabled(self.sr is not None)

    def _refresh_input_preview(self):
        if self.input_image_path is None:
            return

        try:
            bgr = cv2.imread(self.input_image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(self.input_image_path)

            target_w = int(bgr.shape[1] * self._current_scale())
            target_h = int(bgr.shape[0] * self._current_scale())
            upscaled_qimg = upscale_bicubic(self.input_image_path, (target_w, target_h))
            self.comparison.set_input_image(upscaled_qimg)
            self.comparison.set_output_image(None)
        except Exception as e:
            self.status_bar.showMessage(f"Error reading image: {e}")

    def _on_run(self):
        if self.sr is None or self.input_image_path is None:
            self.status_bar.showMessage("Load model and image first.")
            return

        if self._current_overlap() >= self._current_tile():
            self.status_bar.showMessage("Overlap must be smaller than tile size.")
            return

        self._apply_settings_to_loaded_tiler()
        self._set_running_state(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running...")

        self.thread = QThread()
        self.worker = InferenceWorker(self.sr, self.input_image_path)
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _on_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Tile {current}/{total}")

    def _on_finished(self, result_qimg: QImage):
        self.output_qimg = result_qimg
        self.comparison.set_output_image(result_qimg)
        self.status_label.setText("Done")
        self.status_bar.showMessage("Inference complete")
        self._set_running_state(False)
        self.btn_save.setEnabled(True)

    def _on_save(self):
        if self.output_qimg is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save output",
            "output.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)",
        )
        if not file_path:
            return

        if self.output_qimg.save(file_path):
            self.status_bar.showMessage(f"Saved to {file_path}")
        else:
            self.status_bar.showMessage("Failed to save the image.")

    def _on_error(self, msg: str):
        self.status_label.setText("Error")
        self.status_bar.showMessage(f"Error: {msg}")
        self._set_running_state(False)

    def _set_running_state(self, running: bool):
        self.btn_run.setEnabled(
            not running and self.sr is not None and self.input_image_path is not None
        )
        self.btn_load_model.setEnabled(not running)
        self.btn_load_image.setEnabled(not running)
        self.btn_save.setEnabled(not running and self.output_qimg is not None)
        self.tile_spin.setEnabled(not running)
        self.overlap_spin.setEnabled(not running)
        self.scale_spin.setEnabled(not running)
        self.device_id_spin.setEnabled(not running)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    window = MainWindow()
    if windows_version_notice_required():
        message = (
            "DirectML requires Windows 10 version 1903 or newer "
            f"(build {WINDOWS_10_1903_BUILD}+). Detected: {windows_version_text()}."
        )
        QMessageBox.warning(None, "Windows version notice", message)
        window.status_bar.showMessage(message)

    window.resize(1200, 700)
    window.show()
    sys.exit(app.exec())
