from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from sr.tiler import SuperResolutionTiler
from sr.util import bgr_array_to_qimage


class InferenceWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(QImage)
    error = Signal(str)

    def __init__(self, sr: SuperResolutionTiler, image_path: str):
        super().__init__()
        self.sr = sr
        self.image_path = image_path

    @Slot()
    def run(self):
        try:
            def report(current, total):
                self.progress.emit(current, total)

            result_bgr = self.sr.run(
                self.image_path,
                progress_callback=report,
            )
            qimg = bgr_array_to_qimage(result_bgr)
            self.finished.emit(qimg)
        except Exception as e:
            self.error.emit(str(e))
