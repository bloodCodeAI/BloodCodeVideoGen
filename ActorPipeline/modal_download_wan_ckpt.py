from __future__ import annotations

import modal

VOL_PATH = "/vol/out"
MODEL_ID = "Wan-AI/Wan2.1-FLF2V-14B-720P"
TARGET_DIR = f"{VOL_PATH}/models/Wan2.1-FLF2V-14B-720P"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[cli]")
)

volume = modal.Volume.from_name("my-volume")
app = modal.App("download-wan-flf2v-ckpt")


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    timeout=6 * 60 * 60,  # large model, give it time
)
def download_wan_model(hf_token: str = ""):
    from pathlib import Path
    from huggingface_hub import snapshot_download

    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID} to {TARGET_DIR}", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        token=hf_token.strip() or None,
        local_dir_use_symlinks=False,
    )

    volume.commit()
    print(f"Done. Model downloaded to: {TARGET_DIR}", flush=True)


@app.local_entrypoint()
def main(hf_token: str = ""):
    download_wan_model.remote(hf_token=hf_token)
