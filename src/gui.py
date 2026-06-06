import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QWidget, )

from app.platform_check import windows_version_notice_required, WINDOWS_10_1903_BUILD, windows_version_text
from app.version_info import APP_NAME, APP_VERSION
from app.worker import InferenceWorker
from sr.tiler import SuperResolutionTiler
from sr.util import tiler_default, upscale_bicubic
from ui.image_compare import ComparisonWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)

        self.sr = [None, None]
        self.model_names = ["", ""]
        self.active_index = 0

        self.input_image_path: str | None = None

        self.original_qimg: QImage | None = None
        self.result = [None, None]

        self.comparison = ComparisonWidget()
        self.comparison.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")

        self.btn_load_model = QPushButton("Load model")
        self.btn_load_image = QPushButton("Load image")
        self.btn_run = QPushButton("RUN !!!!!")
        self.btn_run.setEnabled(False)
        self.btn_save = QPushButton("Save current")
        self.btn_save.setEnabled(False)
        self.btn_clear_results = QPushButton("Clear all")

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

        self.dual_mode_cb = QCheckBox("Keep slots")
        self.active_model_combo = QComboBox()
        self.active_model_combo.addItems(["Model 1", "Model 2"])
        self.active_model_combo.setEnabled(False)

        self.lbl_model_1 = QLabel("Model 1: –")
        self.lbl_model_2 = QLabel("Model 2: –")
        self.compare_combo = QComboBox()
        self.compare_combo.addItems([
            "Original <> Model 1",
            "Original <> Model 2",
            "Model 1 <> Model 2"
        ])
        self.compare_combo.setEnabled(False)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        model_params_box = QGroupBox("Params")
        model_params_layout = QFormLayout(model_params_box)
        model_params_layout.addRow("Tile size", self.tile_spin)
        model_params_layout.addRow("Overlap", self.overlap_spin)
        model_params_layout.addRow("Scale", self.scale_spin)
        model_params_layout.addRow("Device ID", self.device_id_spin)

        control_box = QGroupBox("Control")
        control_layout = QVBoxLayout(control_box)
        control_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(self.btn_load_image)
        control_layout.addWidget(self.btn_run)
        control_layout.addWidget(self.btn_save)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_label)

        model_mgmt_box = QGroupBox("Slots")
        mgmt_layout = QVBoxLayout(model_mgmt_box)
        mgmt_layout.addWidget(self.dual_mode_cb)
        mgmt_layout.addWidget(QLabel("Active slot:"))
        mgmt_layout.addWidget(self.active_model_combo)
        mgmt_layout.addWidget(self.lbl_model_1)
        mgmt_layout.addWidget(self.lbl_model_2)

        comp_box = QGroupBox("Preview")
        comp_layout = QVBoxLayout(comp_box)
        comp_layout.addWidget(self.compare_combo)
        comp_layout.addWidget(self.btn_clear_results)

        left_layout.addWidget(model_params_box)
        left_layout.addWidget(control_box)
        left_layout.addWidget(model_mgmt_box)
        left_layout.addWidget(comp_box)
        left_layout.addStretch()

        preview_box = QGroupBox("Image preview")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.comparison, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(preview_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 820])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.btn_load_model.clicked.connect(self._on_load_model)
        self.btn_load_image.clicked.connect(self._on_load_image)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_clear_results.clicked.connect(self._on_clear_results)
        self.compare_combo.currentIndexChanged.connect(self._on_compare_mode_changed)
        self.dual_mode_cb.toggled.connect(self._on_dual_mode_toggled)
        self.active_model_combo.currentIndexChanged.connect(self._on_active_model_changed)

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

    def _active_tiler(self) -> SuperResolutionTiler | None:
        return self.sr[self.active_index]

    def _apply_settings_to_active_tiler(self):
        tiler = self._active_tiler()
        if tiler is None:
            return
        tiler.tile = self._current_tile()
        tiler.overlap = self._current_overlap()
        tiler.scale = self._current_scale()
        tiler.device_id = self._current_device_id()

    def _on_dual_mode_toggled(self, checked: bool):
        self.active_model_combo.setEnabled(checked)
        if not checked:
            self.active_model_combo.setCurrentIndex(0)
            self.active_index = 0
            self.sr[1] = None
            self.model_names[1] = ""
            self.result[1] = None
            self.lbl_model_2.setText("Model 2: –")
            self._update_comparison()
        self._update_run_button_state()

    def _on_active_model_changed(self, idx: int):
        self.active_index = idx
        self._apply_settings_to_active_tiler()
        self._update_comparison()
        self._update_run_button_state()

    def _update_run_button_state(self):
        can_run = (
                self._active_tiler() is not None
                and self.input_image_path is not None
        )
        self.btn_run.setEnabled(can_run and not self._is_running)

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX model", "", "ONNX Files (*.onnx);;All Files (*)"
        )
        if not path:
            return

        model_name = Path(path).stem

        if not self.dual_mode_cb.isChecked():
            slot = 0
            self.sr[1] = None
            self.model_names[1] = ""
            self.result[1] = None
            self.lbl_model_2.setText("Model 2: –")
        else:
            slot = self._choose_slot_for_loading()
            if slot is None:
                return

        try:
            tiler = self._build_tiler(path)
            tiler.load()
            self.sr[slot] = tiler
            self.model_names[slot] = model_name

            if slot == 0:
                self.lbl_model_1.setText(f"Model 1: {model_name}")
            else:
                self.lbl_model_2.setText(f"Model 2: {model_name}")

            self.status_bar.showMessage(f"Loaded model into slot {slot + 1}: {model_name}")

            if not self.dual_mode_cb.isChecked():
                self.active_index = 0
                self.active_model_combo.setCurrentIndex(0)
            else:
                pass

            if self.input_image_path:
                self._refresh_input_preview()
            self._update_run_button_state()

        except Exception as e:
            self.status_bar.showMessage(f"Error loading model: {e}")

    def _choose_slot_for_loading(self) -> int | None:
        both_full = self.sr[0] is not None and self.sr[1] is not None

        if not both_full:
            return 0 if self.sr[0] is None else 1

        msg = QMessageBox(self)
        msg.setWindowTitle("Overwrite slot")
        msg.setText("Both slots are full. Choose which model to replace:")
        btn1 = msg.addButton("Overwrite slot 1", QMessageBox.AcceptRole)
        btn2 = msg.addButton("Overwrite slot 2", QMessageBox.AcceptRole)
        msg.exec()

        if msg.clickedButton() == btn1:
            return 0
        elif msg.clickedButton() == btn2:
            return 1
        else:
            return None

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
        self.status_bar.showMessage(f"Input image: {Path(path).name}")
        self._refresh_input_preview()
        self._update_run_button_state()

    def _refresh_input_preview(self):
        if self.input_image_path is None:
            return
        try:
            bgr = cv2.imread(self.input_image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(self.input_image_path)
            target_w = int(bgr.shape[1] * self._current_scale())
            target_h = int(bgr.shape[0] * self._current_scale())
            self.original_qimg = upscale_bicubic(self.input_image_path, (target_w, target_h))
            self._update_comparison()
        except Exception as e:
            self.status_bar.showMessage(f"Error reading image: {e}")

    def _on_run(self):
        tiler = self._active_tiler()
        if tiler is None or self.input_image_path is None:
            self.status_bar.showMessage("Load model and image first.")
            return
        if self._current_overlap() >= self._current_tile():
            self.status_bar.showMessage("Overlap must be smaller than tile size.")
            return

        self._apply_settings_to_active_tiler()
        self._set_running_state(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Running {self.model_names[self.active_index]}...")

        self.thread = QThread()
        self.worker = InferenceWorker(tiler, self.input_image_path)
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
        slot = self.active_index
        self.result[slot] = result_qimg

        if slot == 0:
            self.lbl_model_1.setText(f"Model 1: {self.model_names[0]}")
        else:
            self.lbl_model_2.setText(f"Model 2: {self.model_names[1]}")

        self.compare_combo.setEnabled(True)
        self._update_comparison()

        self.status_label.setText("Done")
        self.status_bar.showMessage(f"Inference complete -> Model {slot + 1}")
        self._set_running_state(False)
        self.btn_save.setEnabled(True)

    def _on_save(self):
        mode = self.compare_combo.currentIndex()
        if mode == 0:
            img = self.result[0]
        elif mode == 1:
            img = self.result[1]
        elif mode == 2:
            img = self.result[1]
        else:
            img = None

        if img is None:
            QMessageBox.information(self, "No result", "No result image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output",
            "output.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)",
        )
        if file_path:
            if img.save(file_path):
                self.status_bar.showMessage(f"Saved to {file_path}")
            else:
                self.status_bar.showMessage("Failed to save the image.")

    def _on_clear_results(self):
        self.result = [None, None]
        self.lbl_model_1.setText(f"Model 1: {self.model_names[0] or '–'}")
        self.lbl_model_2.setText(f"Model 2: {self.model_names[1] or '–'}")
        self.compare_combo.setEnabled(False)
        self.btn_save.setEnabled(False)
        self._update_comparison()
        self.status_bar.showMessage("Results cleared.")

    def _on_compare_mode_changed(self, _):
        self._update_comparison()

    def _update_comparison(self):
        if self.original_qimg is None:
            return

        mode = self.compare_combo.currentIndex()
        left, right = None, None
        left_label, right_label = "", ""

        if mode == 0:
            left, right = self.original_qimg, self.result[0]
            left_label = "Original"
            right_label = self.model_names[0] or "Model 1"
        elif mode == 1:
            left, right = self.original_qimg, self.result[1]
            left_label = "Original"
            right_label = self.model_names[1] or "Model 2"
        elif mode == 2:
            left, right = self.result[0], self.result[1]
            left_label = self.model_names[0] or "Model 1"
            right_label = self.model_names[1] or "Model 2"

        if mode in (0, 1) and right is None:
            right = self.original_qimg
            right_label = "No result yet"
        if mode == 2 and (left is None or right is None):
            left = right = self.original_qimg
            left_label = "Original (need results)"
            right_label = "Original (need results)"

        self.comparison.set_images(left, right, left_label, right_label)

    def _on_settings_changed(self, *args):
        if self.overlap_spin.value() >= self.tile_spin.value():
            self.overlap_spin.setValue(max(0, self.tile_spin.value() - 1))
            return
        self._apply_settings_to_active_tiler()
        if self.input_image_path:
            self._refresh_input_preview()

    def _on_error(self, msg: str):
        self.status_label.setText("Error")
        self.status_bar.showMessage(f"Error: {msg}")
        self._set_running_state(False)

    _is_running = False

    def _set_running_state(self, running: bool):
        self._is_running = running
        self.btn_load_model.setEnabled(not running)
        self.btn_load_image.setEnabled(not running)
        self.btn_save.setEnabled(not running and (self.result[0] is not None or self.result[1] is not None))
        self.btn_clear_results.setEnabled(not running)
        self.dual_mode_cb.setEnabled(not running)
        self.active_model_combo.setEnabled(not running and self.dual_mode_cb.isChecked())
        self.tile_spin.setEnabled(not running)
        self.overlap_spin.setEnabled(not running)
        self.scale_spin.setEnabled(not running)
        self.device_id_spin.setEnabled(not running)
        self._update_run_button_state()


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

    window.resize(1200, 768)
    window.show()
    sys.exit(app.exec())
