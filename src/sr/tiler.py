import time
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

HAT_WINDOW_SIZE = 16
HAT_UPSCALE = 4


def _pad_reflect(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return img

    border_type = (
        cv2.BORDER_REPLICATE if h <= 1 or w <= 1 else cv2.BORDER_REFLECT_101
    )
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=border_type)


class SuperResolutionTiler:
    def __init__(
            self,
            model_path: Union[str, Path],
            tile: int = 112,
            overlap: int = 16,
            scale: int = 4,
            device_id: int = 0,
    ):
        self.model_path = Path(model_path)
        self.tile = tile
        self.overlap = overlap
        self.scale = scale
        self.device_id = device_id

        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None

    def _validate_params(self) -> None:
        if self.tile <= 0:
            raise ValueError("tile must be positive")

        if self.tile % HAT_WINDOW_SIZE != 0:
            raise ValueError(f"tile must be divisible by {HAT_WINDOW_SIZE}")

        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")

        if self.overlap >= self.tile:
            raise ValueError("overlap must be smaller than tile")

        if self.scale != HAT_UPSCALE:
            raise ValueError(f"scale must be {HAT_UPSCALE}")

    def load(self) -> None:
        if self._session is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [
            (
                "DmlExecutionProvider",
                {"device_id": self.device_id},
            ),
            "CPUExecutionProvider",
        ]

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=so,
            providers=providers,
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        print("Loaded model:", self.model_path)
        print("Active providers:", self._session.get_providers())
        print(
            f"Input: {self._input_name}  shape={self._session.get_inputs()[0].shape}"
        )
        print(
            f"Output: {self._output_name}  shape={self._session.get_outputs()[0].shape}"
        )

    def unload(self) -> None:
        self._session = None
        self._input_name = None
        self._output_name = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def run(
            self,
            image: Union[str, Path, np.ndarray],
            output_path: Optional[Union[str, Path]] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        self._validate_params()

        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read: {image}")
        elif isinstance(image, np.ndarray):
            img_bgr = image.copy()
        else:
            raise TypeError("Must be a path or a numpy array.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        t0 = time.perf_counter()
        out_rgb = self._infer_tiled(img_rgb, progress_callback)
        t1 = time.perf_counter()
        print(f"Inference completed in {t1 - t0:.2f}s")

        out_bgr = cv2.cvtColor(
            (out_rgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )

        if output_path is not None:
            cv2.imwrite(str(output_path), out_bgr)
            print(f"Saved output to {output_path}")

        return out_bgr

    def _run_single_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        h, w = tile_rgb.shape[:2]

        if h <= 0 or w <= 0:
            raise ValueError("Tile has invalid size")

        if h % HAT_WINDOW_SIZE != 0 or w % HAT_WINDOW_SIZE != 0:
            raise ValueError(f"Tile height and width must be divisible by {HAT_WINDOW_SIZE}")

        x = np.ascontiguousarray(tile_rgb.transpose(2, 0, 1)[None])
        y = self._session.run([self._output_name], {self._input_name: x})[0]

        sr = y[0].transpose(1, 2, 0)

        expected_h = h * self.scale
        expected_w = w * self.scale
        if sr.shape[0] != expected_h or sr.shape[1] != expected_w:
            raise RuntimeError(
                f"Unexpected size: {sr.shape[1]}x{sr.shape[0]}, "
                f"expected {expected_w}x{expected_h}"
            )

        return sr

    def _make_weight_mask(self, tile_h: int, tile_w: int) -> np.ndarray:
        out_h = tile_h * self.scale
        out_w = tile_w * self.scale
        ov = self.overlap * self.scale

        if ov <= 0:
            return np.ones((out_h, out_w, 1), dtype=np.float32)

        ov_y = min(ov, out_h // 2)
        ov_x = min(ov, out_w // 2)

        wy = np.ones(out_h, dtype=np.float32)
        wx = np.ones(out_w, dtype=np.float32)

        if ov_y > 0:
            ramp_y = np.linspace(0.0, 1.0, ov_y, dtype=np.float32)
            wy[:ov_y] = ramp_y
            wy[-ov_y:] = ramp_y[::-1]

        if ov_x > 0:
            ramp_x = np.linspace(0.0, 1.0, ov_x, dtype=np.float32)
            wx[:ov_x] = ramp_x
            wx[-ov_x:] = ramp_x[::-1]

        mask = wy[:, None] * wx[None, :]
        mask = np.maximum(mask, 1e-3)
        return mask[:, :, None].astype(np.float32)

    def _infer_tiled(
            self,
            img_rgb: np.ndarray,
            progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        h0, w0 = img_rgb.shape[:2]

        if h0 < self.tile or w0 < self.tile:
            img_rgb = _pad_reflect(img_rgb, max(h0, self.tile), max(w0, self.tile))

        h, w = img_rgb.shape[:2]
        stride = self.tile - self.overlap
        if stride <= 0:
            raise ValueError("overlap must be smaller than tile")

        ys = list(range(0, max(h - self.tile, 0) + 1, stride))
        xs = list(range(0, max(w - self.tile, 0) + 1, stride))
        if not ys:
            ys = [0]
        if not xs:
            xs = [0]
        if ys[-1] != h - self.tile:
            ys.append(h - self.tile)
        if xs[-1] != w - self.tile:
            xs.append(w - self.tile)

        out_h = h * self.scale
        out_w = w * self.scale
        acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
        mask = self._make_weight_mask(self.tile, self.tile)

        total = len(ys) * len(xs)
        idx = 0

        for y in ys:
            for x in xs:
                idx += 1
                tile_rgb = img_rgb[y: y + self.tile, x: x + self.tile, :]
                tile_rgb = _pad_reflect(tile_rgb, self.tile, self.tile)

                sr = self._run_single_tile(tile_rgb)

                oy = y * self.scale
                ox = x * self.scale
                th = self.tile * self.scale
                tw = self.tile * self.scale

                acc[oy: oy + th, ox: ox + tw, :] += sr * mask
                weight[oy: oy + th, ox: ox + tw, :] += mask

                if progress_callback is not None:
                    progress_callback(idx, total)

        out = acc / np.maximum(weight, 1e-6)
        out = out[: h0 * self.scale, : w0 * self.scale, :]
        return np.clip(out, 0.0, 1.0)


if __name__ == "__main__":
    MODEL_PATH = "Real_HAT_GAN_SRx4.onnx"
    INPUT_IMG = "test.png"
    OUTPUT_IMG = "test_out.png"


    def progress_cb(current, total):
        print(f"progress: {current}/{total}", end="\r")


    sr = SuperResolutionTiler(
        model_path=MODEL_PATH,
        tile=112,
        overlap=16,
        scale=4,
        device_id=0,
    )

    sr.load()
    result = sr.run(
        image=INPUT_IMG,
        output_path=OUTPUT_IMG,
        progress_callback=progress_cb,
    )

    sr.unload()
