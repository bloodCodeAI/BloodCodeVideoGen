from __future__ import annotations

import modal

WAN_REPO_DIR = "/opt/Wan2.1"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "ninja",
        "packaging",
        "psutil",
        "easydict",
        "diffusers", 
        "ftfy",
        "transformers",
    )
    .run_commands(
        f"git clone https://github.com/Wan-Video/Wan2.1.git {WAN_REPO_DIR}",
        "python -m pip install --upgrade pip setuptools wheel",
        # Fail-fast compile in CUDA devel image (has nvcc).
        "python -m pip install flash-attn --no-build-isolation",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "8",
        }
    )
)

app = modal.App("diag-flashattn")


@app.function(image=image, gpu="T4", timeout=30 * 60)
def check():
    import sys
    import torch

    print("python:", sys.version, flush=True)
    print("torch:", torch.__version__, flush=True)
    print("torch_cuda:", torch.version.cuda, flush=True)
    print("cuda_available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
        print("capability:", torch.cuda.get_device_capability(0), flush=True)

    import flash_attn  # noqa: F401
    print("flash_attn import: OK", flush=True)

    sys.path.insert(0, WAN_REPO_DIR)
    import wan.modules.attention as attn

    print("WAN FLASH_ATTN_2_AVAILABLE:", attn.FLASH_ATTN_2_AVAILABLE, flush=True)


@app.local_entrypoint()
def main():
    check.remote()
