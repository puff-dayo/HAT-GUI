# HAT GUI

A lightweight Windows GUI app for HAT model inference using Microsoft DirectML on DirectX 12. The GUI is built with PySide6, and the binary is packaged with PyInstaller.

There is no need to download or configure Torch, cuDNN, or CUDA with this app. It is portable; simply unzip it to run it.

Currently, it supports Real_HAT_GAN_SRx4 and Real_HAT_GAN_SRx4_sharper. **Real_HAT_GAN_SRx4** would have much better fidelity. Converted ONNX models and the convert script can be downloaded inside [/models](models) folder.

[//]: # (![demo]&#40;assets/scrsht.png&#41;)

## Requirements

Windows 10 version 1903+  and a DirectX 12 compatible GPU.

Tile size 112 may be using 2 gigabytes of VRAM and 160 using 3GB, but a larger tile size does not mean to be faster even with VRAM is at mid or low usage. Have fun testing with different params.

## Installation

Check Release for _untrusted_ binary builds, or run from sauce with `uv sync` and `uv run .\src\gui.py`. To build, run `build_msw.bat` and manually remove the unexpectedly bundled ffmpeg.

## Acknowledgements

The original work and pretrained models come from the following papers.

Activating More Pixels in Image Super-Resolution Transformer (CVPR 2023). [[Paper Link]](https://arxiv.org/abs/2205.04437)<br>
HAT: Hybrid Attention Transformer for Image Restoration (arXiv 2023) [[Paper Link]](https://arxiv.org/abs/2309.05239).

The original HAT code and repo: [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT). A huge thank-you for NeoChen1024's fork for modern Python/PyTorch fixes on [NeoChen1024/HAT-f](https://github.com/NWeoChen1024/HAT-f).