import argparse
from pathlib import Path

# I'm using torch 2.11.0+cu130, onnx 1.21.0 on MSW11 with Python 3.13
import torch
import torch.nn as nn

# For modern Python and torch, try to use this fork of HAT.
# https://github.com/NeoChen1024/HAT-f
from hat.archs.hat_arch import HAT


class RealHATWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def build_real_hat_gan_srx4():
    return HAT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def load_hat_weights(model: nn.Module, weight_path: str, param_key: str = "params_ema"):
    ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and param_key in ckpt:
        state = ckpt[param_key]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        state = ckpt["params"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=True)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="Real_HAT_GAN_SRx4.pth")
    parser.add_argument("--out", default="real_hat_gan_srx4_tile128.onnx")
    parser.add_argument("--opset", type=int, default=21)
    parser.add_argument("--trace-size", type=int, default=112)
    args = parser.parse_args()

    assert args.trace_size % 16 == 0, "export-size must be divisible by 16."

    model = build_real_hat_gan_srx4()
    load_hat_weights(model, args.weights, param_key="params_ema")
    model.eval()

    wrapped = RealHATWrapper(model).eval()

    x = torch.rand(
        1,
        3,
        args.trace_size,
        args.trace_size,
        dtype=torch.float32,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        y = wrapped(x)
        print("torch output:", y.shape, y.dtype, float(y.min()), float(y.max()))

        torch.onnx.export(
            wrapped,
            x,
            str(out_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {2: "height", 3: "width"},
                "output": {2: "out_height", 3: "out_width"},
            },
            dynamo=False,
        )

    print("ONNX saved:", out_path)


if __name__ == "__main__":
    main()

# For example, we can run
# uv run python convert.py `
#   --weights Real_HAT_GAN_SRx4.pth `
#   --out Real_HAT_GAN_SRx4.onnx `
#   --trace-size 112 `
#   --opset 20
#
# Oh and DON'T try to quantize HAT models to even fp16.
# It damages the quality a lot even with subtle quantization.
