from PySide6.QtCore import Qt, QRectF, QPointF, QSizeF, Signal
from PySide6.QtGui import QPainter, QImage, QPen, QColor, QBrush
from PySide6.QtWidgets import QWidget


class ComparisonWidget(QWidget):
    split_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_image: QImage | None = None
        self._output_image: QImage | None = None
        self._split = 0.5
        self._dragging = False
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

    def set_input_image(self, qimg: QImage):
        self._input_image = qimg
        self.update()

    def set_output_image(self, qimg: QImage):
        self._output_image = qimg
        self.update()

    def set_images(self, left: QImage, right: QImage, left_label: str = "", right_label: str = ""):
        self._input_image = left
        self._output_image = right
        self._left_label = left_label
        self._right_label = right_label
        self.update()

    def set_split(self, fraction: float):
        self._split = max(0.0, min(1.0, fraction))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._input_image is None and self._output_image is None:
            painter.fillRect(self.rect(), QColor(50, 50, 50))
            return

        img_rect = self._image_rect()
        if img_rect.isEmpty():
            return

        if self._input_image:
            left_clip = QRectF(
                img_rect.left(), img_rect.top(),
                self._split * img_rect.width(), img_rect.height()
            )
            painter.save()
            painter.setClipRect(left_clip)
            painter.drawImage(img_rect, self._input_image)
            painter.restore()

        if self._output_image:
            right_clip = QRectF(
                img_rect.left() + self._split * img_rect.width(),
                img_rect.top(),
                (1 - self._split) * img_rect.width(), img_rect.height()
            )
            painter.save()
            painter.setClipRect(right_clip)
            painter.drawImage(img_rect, self._output_image)
            painter.restore()

        line_x = img_rect.left() + self._split * img_rect.width()
        pen = QPen(QColor(255, 128, 128), 2)
        painter.setPen(pen)
        painter.drawLine(
            QPointF(line_x, img_rect.top()),
            QPointF(line_x, img_rect.bottom())
        )

        handle_radius = 10
        handle_center = QPointF(line_x, img_rect.center().y())
        painter.setBrush(QBrush(QColor(255, 128, 128)))
        painter.drawEllipse(handle_center, handle_radius, handle_radius)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_near_split(event.pos()):
            self._dragging = True
            self.setCursor(Qt.SplitHCursor)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_split_from_x(event.pos().x())
        else:
            self.setCursor(
                Qt.SplitHCursor if self._is_near_split(event.pos()) else Qt.ArrowCursor
            )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)

    def _image_rect(self) -> QRectF:
        if self._input_image is not None:
            src = self._input_image.size()
        elif self._output_image is not None:
            src = self._output_image.size()
        else:
            return QRectF()

        src_size = QSizeF(src)
        widget_size = QSizeF(self.rect().size())
        scaled_size = src_size.scaled(widget_size, Qt.KeepAspectRatio)
        r = QRectF(QPointF(0, 0), scaled_size)
        r.moveCenter(self.rect().center())
        return r

    def _is_near_split(self, pos: QPointF) -> bool:
        img_rect = self._image_rect()
        if img_rect.isEmpty():
            return False
        split_x = img_rect.left() + self._split * img_rect.width()
        return abs(pos.x() - split_x) < 15 and img_rect.contains(pos)

    def _update_split_from_x(self, mouse_x: float):
        img_rect = self._image_rect()
        if img_rect.width() == 0:
            return
        rel_x = (mouse_x - img_rect.left()) / img_rect.width()
        self._split = max(0.0, min(1.0, rel_x))
        self.split_changed.emit(self._split)
        self.update()