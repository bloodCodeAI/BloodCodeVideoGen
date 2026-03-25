"""
Train an SDXL LoRA for actor identity on Modal (H100).

This script expects actor photos to exist locally in:
  BloodCodeVideoGen/ActorImages/

It uploads those images to the remote runner, launches the official
Diffusers DreamBooth LoRA SDXL training script, then saves the weights to:
  my-volume/actor_pipeline/lora/<run_id>/

Usage:
  modal run ActorPipeline/modal_train_actor_lora.py
  modal run ActorPipeline/modal_train_actor_lora.py --max-train-steps 1200 --trigger-token sksactor
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

CACHE_DIR = "/cache"
OUTPUT_DIR = "/vol/out"
SUBFOLDER = "actor_pipeline/lora"
MINUTES = 60

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"

LOCAL_ACTOR_DIR = Path("c:/Python/BloodCodeAi/BloodCodeVideoGen/ActorImages")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=0.33.0",
        "bitsandbytes",
        "datasets",
        "huggingface_hub",
        "peft>=0.12.0",
        "Pillow",
        "safetensors",
        "torch>=2.5.0",
        "torchvision",
        "transformers>=4.44.0",
    )
    .run_commands(
        "git clone https://github.com/huggingface/diffusers /opt/diffusers",
        "python -m pip install -e /opt/diffusers",
    )
    .env({"HF_HUB_CACHE": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

cache_vol = modal.Volume.from_name("video-gen-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("my-volume")
app = modal.App("actor-lora-train")


@app.cls(
    image=image,
    gpu="H100",
    timeout=180 * MINUTES,
    volumes={CACHE_DIR: cache_vol, OUTPUT_DIR: output_vol},
)
class LoraTrainer:
    @modal.method()
    def train(
        self,
        actor_images: list[tuple[str, bytes]],
        run_id: str,
        trigger_token: str = "sksactor",
        max_train_steps: int = 1000,
        learning_rate: float = 1e-4,
    ) -> dict:
        t_total = time.time()
        work = Path("/tmp/actor_lora")
        instance_dir = work / "instance"
        out_dir = work / "output"
        instance_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, blob in actor_images:
            (instance_dir / name).write_bytes(blob)

        instance_prompt = f"a photo of {trigger_token} person"
        cmd = [
            "python",
            "/opt/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py",
            "--pretrained_model_name_or_path",
            BASE_MODEL,
            "--pretrained_vae_model_name_or_path",
            VAE_MODEL,
            "--instance_data_dir",
            str(instance_dir),
            "--output_dir",
            str(out_dir),
            "--instance_prompt",
            instance_prompt,
            "--resolution",
            "1024",
            "--train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "4",
            "--learning_rate",
            str(learning_rate),
            "--lr_scheduler",
            "constant",
            "--lr_warmup_steps",
            "0",
            "--max_train_steps",
            str(max_train_steps),
            "--mixed_precision",
            "bf16",
            "--rank",
            "16",
            "--seed",
            "42",
            "--gradient_checkpointing",
            "--use_8bit_adam",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            captured.append(line)
            if len(captured) > 500:
                captured = captured[-500:]
        ret = proc.wait()
        if ret != 0:
            tail = "".join(captured[-200:])
            raise RuntimeError(f"LoRA training failed (exit={ret}).\nOutput tail:\n{tail}")

        volume_dir = Path(OUTPUT_DIR) / SUBFOLDER / run_id
        volume_dir.mkdir(parents=True, exist_ok=True)
        for p in out_dir.glob("*"):
            if p.is_file():
                (volume_dir / p.name).write_bytes(p.read_bytes())

        meta = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_model": BASE_MODEL,
            "vae_model": VAE_MODEL,
            "trigger_token": trigger_token,
            "instance_prompt": instance_prompt,
            "num_actor_images": len(actor_images),
            "max_train_steps": max_train_steps,
            "learning_rate": learning_rate,
            "training_wall_s": round(time.time() - t_total, 2),
            "weights_dir": str(volume_dir),
        }
        (volume_dir / "train_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        output_vol.commit()
        return meta


@app.local_entrypoint()
def main(
    trigger_token: str = "sksactor",
    max_train_steps: int = 1000,
    learning_rate: float = 1e-4,
):
    if not LOCAL_ACTOR_DIR.exists():
        raise FileNotFoundError(f"Missing actor image folder: {LOCAL_ACTOR_DIR}")

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted(
        p
        for p in LOCAL_ACTOR_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )
    if len(files) < 8:
        raise ValueError(
            f"Found only {len(files)} actor images. Need at least 8 (recommended 20-30)."
        )

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S") + "_actor_lora"
    blobs = [(p.name, p.read_bytes()) for p in files]
    print(f"Training actor LoRA on {len(blobs)} images...")
    print(f"Run ID: {run_id}")
    print(f"Trigger token: {trigger_token}")
    print(f"Output: my-volume/{SUBFOLDER}/{run_id}/")

    meta = LoraTrainer().train.remote(
        actor_images=blobs,
        run_id=run_id,
        trigger_token=trigger_token,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
    )
    print("Done.")
    print(json.dumps(meta, indent=2))
