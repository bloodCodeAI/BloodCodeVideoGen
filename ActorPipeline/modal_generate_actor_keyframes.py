"""
Generate consistent actor keyframes with:
  SDXL + ControlNet(OpenPose) + IP-Adapter + trained LoRA.

Inputs expected locally:
  - Actor reference image(s): BloodCodeVideoGen/ActorImages/
  - Pose images: BloodCodeVideoGen/posesImages/standardized_png_1024/

LoRA weights expected on volume from training script:
  my-volume/actor_pipeline/lora/<run_id>/

Usage:
  modal run ActorPipeline/modal_generate_actor_keyframes.py --lora-run-id <run_id>
  modal run ActorPipeline/modal_generate_actor_keyframes.py --lora-run-id <run_id> --ip-adapter-scale 0.65 --controlnet-scale 0.9
  modal run ActorPipeline/modal_generate_actor_keyframes.py --lora-run-id <run_id> --trigger-token sksactor
  modal run ActorPipeline/modal_generate_actor_keyframes.py --lora-run-id <run_id> --scenario-path ActorPipeline/squatt_scenario.json
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

CACHE_DIR = "/cache"
OUTPUT_DIR = "/vol/out"
SUBFOLDER = "actor_pipeline/keyframes"
LORA_SUBFOLDER = "actor_pipeline/lora"
MINUTES = 60

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL_ID = "thibaud/controlnet-openpose-sdxl-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"

LOCAL_ACTOR_DIR = Path("c:/Python/BloodCodeAi/BloodCodeVideoGen/ActorImages")
LOCAL_POSE_DIR = Path("c:/Python/BloodCodeAi/BloodCodeVideoGen/posesImages/standardized_png_1024")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate>=0.33.0",
        "controlnet-aux",
        "diffusers>=0.36.0",
        "peft>=0.12.0",
        "Pillow",
        "safetensors",
        "torch>=2.5.0",
        "transformers>=4.44.0",
    )
    .env({"HF_HUB_CACHE": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

cache_vol = modal.Volume.from_name("video-gen-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("my-volume")
app = modal.App("actor-keyframe-gen")


def _default_prompt(i: int) -> str:
    prompts = [
        "a fit person standing neutral in a bright gym, full body, photorealistic",
        "a fit person raising both arms overhead in a bright gym, full body, photorealistic",
        "a fit person doing a jumping jack in a bright gym, full body, photorealistic",
        "a fit person in squat bottom position in a bright gym, full body, photorealistic",
        "a fit person in squat middle position in a bright gym, full body, photorealistic",
        "a fit person standing up from squat in a bright gym, full body, photorealistic",
        "a fit person in left lunge in a bright gym, full body, photorealistic",
        "a fit person in right lunge in a bright gym, full body, photorealistic",
        "a fit person standing neutral finish pose in a bright gym, full body, photorealistic",
    ]
    return prompts[i % len(prompts)]


DEFAULT_POSE_ORDER = [
    "pose_007.png",  # neutral
    "pose_001.png",  # arms over head
    "pose_003.png",  # jumping jack
    "pose_013.png",  # squat down
    "pose_014.png",  # squat mid
    "pose_015.png",  # squat up
    "pose_005.png",  # lunge left
    "pose_006.png",  # lunge right
    "pose_007.png",  # neutral finish
]


def _load_scenario(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Scenario JSON root must be an object.")
    if "steps" not in data or not isinstance(data["steps"], list) or not data["steps"]:
        raise ValueError("Scenario JSON must include non-empty 'steps' list.")
    return data


def _resolve_scenario_poses(
    scenario: dict,
    scenario_path: Path,
) -> tuple[list[Path], list[str], str | None, str | None]:
    pose_root = scenario.get("pose_root_dir")
    if pose_root:
        root = Path(pose_root)
    else:
        root = scenario_path.parent
    if not root.is_absolute():
        root = (scenario_path.parent / root).resolve()

    poses = scenario.get("poses", {})
    if poses is not None and not isinstance(poses, dict):
        raise ValueError("'poses' in scenario JSON must be an object if present.")

    steps = scenario["steps"]
    sorted_steps = sorted(
        steps,
        key=lambda x: x.get("k", 0) if isinstance(x, dict) else 0,
    )
    pose_files: list[Path] = []
    prompts: list[str] = []

    for idx, step in enumerate(sorted_steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step #{idx} must be an object.")

        pose_file: Path | None = None
        pose_id = step.get("pose_id")
        pose_rel = step.get("pose_file")
        if pose_id:
            if not poses or pose_id not in poses:
                raise ValueError(f"Step #{idx}: pose_id '{pose_id}' not found in scenario 'poses'.")
            pose_meta = poses[pose_id]
            if not isinstance(pose_meta, dict) or "file" not in pose_meta:
                raise ValueError(f"Step #{idx}: pose_id '{pose_id}' must map to object with 'file'.")
            pose_rel = pose_meta["file"]
        if not pose_rel:
            raise ValueError(f"Step #{idx}: provide either 'pose_id' or 'pose_file'.")

        pose_file = (root / str(pose_rel)).resolve()
        if not pose_file.exists():
            raise FileNotFoundError(f"Step #{idx}: pose file not found: {pose_file}")
        pose_files.append(pose_file)

        prompt = step.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Step #{idx}: missing non-empty 'prompt'.")
        prompts.append(prompt.strip())

    global_style = scenario.get("global_style")
    if isinstance(global_style, str):
        global_style = global_style.strip() or None
    else:
        global_style = None

    negative_prompt = scenario.get("negative_prompt")
    if isinstance(negative_prompt, str):
        negative_prompt = negative_prompt.strip() or None
    else:
        negative_prompt = None

    return pose_files, prompts, global_style, negative_prompt


@app.cls(
    image=image,
    gpu="H100",
    timeout=90 * MINUTES,
    volumes={CACHE_DIR: cache_vol, OUTPUT_DIR: output_vol},
)
class KeyframeGenerator:
    @modal.enter()
    def load(self):
        import torch
        from controlnet_aux import OpenposeDetector
        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
        )

        t0 = time.time()
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
            variant="fp16",
        ).to("cuda")
        self.pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT,
        )
        self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        self.torch = torch
        self.cold_start_s = round(time.time() - t0, 2)

    @modal.method()
    def generate(
        self,
        run_id: str,
        lora_run_id: str,
        trigger_token: str,
        actor_reference: tuple[str, bytes],
        pose_images: list[tuple[str, bytes]],
        prompt_sequence: list[str] | None = None,
        global_style: str | None = None,
        negative_prompt: str | None = None,
        ip_adapter_scale: float = 0.65,
        controlnet_scale: float = 0.9,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
    ) -> dict:
        from PIL import Image

        lora_dir = Path(OUTPUT_DIR) / LORA_SUBFOLDER / lora_run_id
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA directory not found on volume: {lora_dir}")

        self.pipe.load_lora_weights(str(lora_dir))
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        base = Path(OUTPUT_DIR) / SUBFOLDER / run_id
        (base / "poses").mkdir(parents=True, exist_ok=True)
        (base / "keyframes").mkdir(parents=True, exist_ok=True)

        ref_name, ref_blob = actor_reference
        ref_img = Image.open(io.BytesIO(ref_blob)).convert("RGB").resize((1024, 1024))
        ref_img.save(base / "actor_reference.png")

        timings = []
        used_prompts: dict[str, str] = {}
        neg = negative_prompt or "blurry, low quality, extra limbs, deformed, cartoon, illustration"
        print(f"Starting keyframe generation for {len(pose_images)} poses", flush=True)

        for i, (pose_name, pose_blob) in enumerate(pose_images, 1):
            pose_img = Image.open(io.BytesIO(pose_blob)).convert("RGB")
            pose_cond = self.pose_detector(pose_img)
            pose_cond = pose_cond.resize((1024, 1024))
            pose_cond.save(base / "poses" / f"pose_{i:03d}.png")

            if prompt_sequence and i - 1 < len(prompt_sequence):
                base_prompt = prompt_sequence[i - 1]
            else:
                base_prompt = _default_prompt(i - 1)
            if global_style:
                base_prompt = f"{base_prompt}, {global_style}"

            prompt = f"{base_prompt}, a photo of {trigger_token} person"
            used_prompts[pose_name] = prompt
            print(
                f"[{i}/{len(pose_images)}] Generating keyframe_{i:03d}.png from pose '{pose_name}'",
                flush=True,
            )
            print(f"[{i}/{len(pose_images)}] Prompt: {prompt}", flush=True)
            t0 = time.time()
            out = self.pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=pose_cond,
                ip_adapter_image=ref_img,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            dt = round(time.time() - t0, 2)
            print(f"[{i}/{len(pose_images)}] Done keyframe_{i:03d}.png in {dt:.2f}s", flush=True)
            timings.append(dt)
            out.save(base / "keyframes" / f"keyframe_{i:03d}.png")

        keyframes_rel = [f"keyframes/keyframe_{i:03d}.png" for i in range(1, len(pose_images) + 1)]
        transitions = [
            {
                "from": keyframes_rel[i],
                "to": keyframes_rel[i + 1],
                "segment_idx": i + 1,
            }
            for i in range(len(keyframes_rel) - 1)
        ]

        meta = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "lora_run_id": lora_run_id,
            "trigger_token": trigger_token,
            "actor_reference": ref_name,
            "num_pose_images": len(pose_images),
            "pose_names": [name for name, _ in pose_images],
            "cold_start_s": self.cold_start_s,
            "ip_adapter_scale": ip_adapter_scale,
            "controlnet_scale": controlnet_scale,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": neg,
            "global_style": global_style,
            "timing": {
                "per_keyframe_s": timings,
                "generation_total_s": round(sum(timings), 2),
                "avg_keyframe_s": round(sum(timings) / max(len(timings), 1), 2),
            },
            "prompts": used_prompts,
            "hunyuan_ready": {
                "ordered_keyframes": keyframes_rel,
                "transitions": transitions,
            },
        }
        (base / "keyframe_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (base / "keyframes_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "ordered_keyframes": keyframes_rel,
                    "transitions": transitions,
                    "notes": "Use each transition pair as Ki -> Ki+1 segment input for your Hunyuan composition stage.",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        output_vol.commit()
        return meta


@app.local_entrypoint()
def main(
    lora_run_id: str,
    trigger_token: str = "sksactor",
    use_default_pose_order: bool = True,
    scenario_path: str = "",
    ip_adapter_scale: float = 0.65,
    controlnet_scale: float = 0.9,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
):
    if not LOCAL_ACTOR_DIR.exists():
        raise FileNotFoundError(f"Missing actor image folder: {LOCAL_ACTOR_DIR}")
    if not LOCAL_POSE_DIR.exists():
        raise FileNotFoundError(f"Missing pose image folder: {LOCAL_POSE_DIR}")

    actor_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jfif"}
    pose_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jfif"}
    actor_files = sorted(
        p
        for p in LOCAL_ACTOR_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in actor_exts
    )
    if not actor_files:
        raise ValueError("No actor images found in ActorImages/")

    scenario_prompts: list[str] | None = None
    scenario_style: str | None = None
    scenario_negative_prompt: str | None = None

    if scenario_path:
        scenario_file = Path(scenario_path)
        scenario = _load_scenario(scenario_file)
        pose_files, scenario_prompts, scenario_style, scenario_negative_prompt = _resolve_scenario_poses(
            scenario,
            scenario_file,
        )
    else:
        pose_files = sorted(
            p
            for p in LOCAL_POSE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in pose_exts
        )
        if not pose_files:
            raise ValueError("No standardized pose images found.")

        if use_default_pose_order:
            by_name = {p.name: p for p in pose_files}
            ordered = []
            for n in DEFAULT_POSE_ORDER:
                if n in by_name:
                    ordered.append(by_name[n])
            if len(ordered) >= 4:
                pose_files = ordered
            else:
                print("Default pose order had too few matches; falling back to sorted pose files.")

    # First actor image is the identity reference for IP-Adapter.
    actor_ref = (actor_files[0].name, actor_files[0].read_bytes())
    poses = [(p.name, p.read_bytes()) for p in pose_files]

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S") + "_actor_keyframes"
    print(f"Generating actor keyframes with LoRA run: {lora_run_id}")
    print(f"Trigger token: {trigger_token}")
    if scenario_path:
        print(f"Scenario: {scenario_path}")
    print(f"Pose count: {len(poses)}")
    print(f"Output: my-volume/{SUBFOLDER}/{run_id}/")
    meta = KeyframeGenerator().generate.remote(
        run_id=run_id,
        lora_run_id=lora_run_id,
        trigger_token=trigger_token,
        actor_reference=actor_ref,
        pose_images=poses,
        prompt_sequence=scenario_prompts,
        global_style=scenario_style,
        negative_prompt=scenario_negative_prompt,
        ip_adapter_scale=ip_adapter_scale,
        controlnet_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    print("Done.")
    print(json.dumps(meta, indent=2))
