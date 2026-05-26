"""Convert sapiens2_0.1b_fp32.onnx → sapiens2_0.1b_fp16.onnx for WebGPU inference.

Usage:
    pip install onnx onnxmltools huggingface_hub
    python scripts/convert_fp16.py [--upload]

The fp32 source is downloaded automatically from HuggingFace if not present locally.
Pass --upload to push the result to barakplasma/sapiens2-onnx.
"""
import sys
from pathlib import Path

SRC = Path("sapiens2_0.1b_fp32.onnx")
DST = Path("sapiens2_0.1b_fp16.onnx")
REPO = "barakplasma/sapiens2-onnx"
FP32_URL = f"https://huggingface.co/{REPO}/resolve/main/sapiens2_0.1b_fp32.onnx"

def download_fp32():
    print(f"Downloading {FP32_URL} ...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=REPO, filename="sapiens2_0.1b_fp32.onnx")
    import shutil
    shutil.copy(path, SRC)
    print(f"  → {SRC} ({SRC.stat().st_size / 1e6:.0f} MB)")

def convert():
    import onnx
    from onnxmltools.utils.float16_converter import convert_float_to_float16

    print(f"Loading {SRC} ...")
    model = onnx.load(str(SRC))

    print("Converting to fp16 (keep_io_types=True) ...")
    # keep_io_types=True preserves float32 inputs/outputs so the browser JS
    # needs no changes — only internal weights/activations become fp16.
    fp16 = convert_float_to_float16(model, keep_io_types=True)

    onnx.save(fp16, str(DST))
    print(f"Saved {DST} ({DST.stat().st_size / 1e6:.0f} MB)")

def upload():
    from huggingface_hub import HfApi
    print(f"Uploading {DST} to {REPO} ...")
    HfApi().upload_file(
        path_or_fileobj=str(DST),
        path_in_repo="sapiens2_0.1b_fp16.onnx",
        repo_id=REPO,
    )
    print("Upload complete.")

if __name__ == "__main__":
    do_upload = "--upload" in sys.argv

    if not SRC.exists():
        download_fp32()

    convert()

    if do_upload:
        upload()
    else:
        print(f"\nRun with --upload to push {DST} to HuggingFace.")
