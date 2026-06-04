import inspect

import cv2
import numpy as np
from PySide6.QtGui import QImage

from .tiler import SuperResolutionTiler


def tiler_default(name: str):
    signature = inspect.signature(SuperResolutionTiler.__init__)
    return signature.parameters[name].default


def bgr_array_to_qimage(bgr: np.ndarray) -> QImage:
    h, w, ch = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return qimg.copy()


def upscale_bicubic(img_path: str, target_size: tuple[int, int]) -> QImage:
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    upscaled = cv2.resize(bgr, target_size, interpolation=cv2.INTER_CUBIC)
    return bgr_array_to_qimage(upscaled)