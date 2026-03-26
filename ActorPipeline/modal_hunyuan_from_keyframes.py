"""
Create a sequential Hunyuan I2V video from pre-generated actor keyframes.

Reads keyframes from Modal volume:
  my-volume/actor_pipeline/keyframes/<keyframe_run_id>/keyframes/keyframe_*.png

Runs sequential I2V segments (one per transition Ki -> K{i+1}), concatenates
the clips, and stores outputs under:
  my-volume/actor_pipeline/hunyuan_video/<video_run_id>/

Usage:
  modal run ActorPipeline/modal_hunyuan_from_keyframes.py --keyframe-run-id <run_id>
  modal run ActorPipeline/modal_hunyuan_from_keyframes.py --keyframe-run-id <run_id> --i2v-steps 20 --num-frames 81
  modal run ActorPipeline/modal_hunyuan_from_keyframes.py --keyframe-run-id <run_id> --scenario-path ActorPipeline/squatt_scenario.json
  modal run ActorPipeline/modal_hunyuan_from_keyframes.py --keyframe-run-id <run_id> --scenario-path ActorPipeline/squatt_scenario.json --lock-end-keyframe
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

CACHE_DIR = "/cache"
OUTPUT_DIR = "/vol/out"
MINUTES = 60

KEYFRAME_SUBFOLDER = "actor_pipeline/keyframes"
VIDEO_SUBFOLDER = "actor_pipeline/hunyuan_video"

HUNYUAN_I2V_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"

DEFAULT_NUM_FRAMES = 81
DEFAULT_I2V_STEPS = 20
DEFAULT_FPS = 15

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=0.33.0",
        "diffusers>=0.36.0",
        "ftfy",
        "imageio[ffmpeg]",
        "Pillow",
        "sentencepiece",
        "torch>=2.5.0",
        "torchvision",
        "transformers>=4.44.0",
    )
    .env({"HF_HUB_CACHE": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with image.imports():
    import torch
    from diffusers.utils import export_to_video

cache_volume = modal.Volume.from_name("video-gen-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("my-volume")
app = modal.App("actor-hunyuan-from-keyframes")


def _frames_to_pil_list(out, np_module, PILImage):
    if isinstance(out, np_module.ndarray):
        return [PILImage.fromarray((f * 255).astype("uint8")) for f in out]
    return list(out)


def _default_transition_prompt(i: int) -> str:
    prompts = [
        "A fitness instructor smoothly transitions into the next exercise pose, cinematic gym lighting, natural motion.",
        "The same fitness instructor performs a controlled transition to the next movement, realistic motion.",
        "The same fitness instructor continues the workout sequence with fluid body movement, photorealistic.",
        "The same fitness instructor changes stance into the next pose with stable camera and natural movement.",
        "The same fitness instructor performs the next exercise transition, clean form, realistic motion.",
        "The same fitness instructor moves into the next workout position, smooth and controlled.",
        "The same fitness instructor transitions to the final pose of the short routine, photorealistic.",
    ]
    return prompts[i % len(prompts)]


def _load_transition_prompts_from_scenario(scenario_path: Path) -> list[str]:
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {scenario_path}")

    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Scenario JSON root must be an object.")
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("Scenario JSON must include at least 2 steps.")

    sorted_steps = sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda x: x.get("k", 0),
    )
    if len(sorted_steps) < 2:
        raise ValueError("Scenario JSON must include at least 2 valid step objects.")

    style = data.get("global_style")
    style_suffix = f" {style}" if isinstance(style, str) and style.strip() else ""

    transition_prompts: list[str] = []
    for i in range(len(sorted_steps) - 1):
        a = str(sorted_steps[i].get("prompt", "")).strip()
        b = str(sorted_steps[i + 1].get("prompt", "")).strip()
        if not a or not b:
            raise ValueError(
                f"Scenario step transition {i + 1}->{i + 2} missing non-empty prompt."
            )
        transition_prompts.append(
            "A fitness instructor transitions smoothly and realistically from the current pose to the next pose. "
            f"Start state: {a}. End state: {b}.{style_suffix}"
        )
    return transition_prompts


@app.cls(
    image=image,
    gpu="H100",
    timeout=120 * MINUTES,
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
)
class HunyuanFromKeyframes:
    @modal.enter()
    def load(self):
        from diffusers import HunyuanVideo15ImageToVideoPipeline
        import PIL.Image as PILImage
        import numpy as np

        t0 = time.time()
        self.pipe_i2v = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
            HUNYUAN_I2V_MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe_i2v.vae.enable_tiling()
        self.PILImage = PILImage
        self.np = np
        self.cold_start_s = round(time.time() - t0, 2)

    @modal.method()
    def run(
        self,
        keyframe_run_id: str,
        video_run_id: str,
        scenario_transition_prompts: list[str] | None = None,
        lock_end_keyframe: bool = True,
        end_blend_frames: int = 6,
        num_frames: int = DEFAULT_NUM_FRAMES,
        i2v_steps: int = DEFAULT_I2V_STEPS,
        fps: int = DEFAULT_FPS,
    ) -> dict:
        import imageio.v3 as iio

        output_volume.reload()
        keyframe_base = Path(OUTPUT_DIR) / KEYFRAME_SUBFOLDER / keyframe_run_id
        keyframe_dir = keyframe_base / "keyframes"
        if not keyframe_dir.exists():
            raise FileNotFoundError(f"Keyframe folder not found: {keyframe_dir}")

        keyframes = sorted(keyframe_dir.glob("keyframe_*.png"))
        if len(keyframes) < 2:
            raise ValueError(f"Need at least 2 keyframes, found {len(keyframes)}")

        out_base = Path(OUTPUT_DIR) / VIDEO_SUBFOLDER / video_run_id
        out_base.mkdir(parents=True, exist_ok=True)

        segment_paths = []
        segment_timings = []
        transition_prompts = []

        for i in range(len(keyframes) - 1):
            start_img = self.PILImage.open(keyframes[i]).convert("RGB")
            end_img = self.PILImage.open(keyframes[i + 1]).convert("RGB")
            if scenario_transition_prompts and i < len(scenario_transition_prompts):
                prompt = scenario_transition_prompts[i]
            else:
                prompt = _default_transition_prompt(i)
            transition_prompts.append(prompt)
            print(
                f"[{i + 1}/{len(keyframes) - 1}] Generating segment from {keyframes[i].name} to {keyframes[i + 1].name}",
                flush=True,
            )
            print(f"[{i + 1}/{len(keyframes) - 1}] Prompt: {prompt}", flush=True)

            t_seg = time.time()
            out = self.pipe_i2v(
                prompt=prompt,
                image=start_img,
                num_frames=num_frames,
                num_inference_steps=i2v_steps,
                output_type="pil",
            ).frames[0]
            dt = round(time.time() - t_seg, 2)
            segment_timings.append(dt)
            print(f"  Segment {i + 1}/{len(keyframes) - 1}: {dt:.1f}s", flush=True)

            out_list = _frames_to_pil_list(out, self.np, self.PILImage)
            # Endpoint lock: force the clip to land on the next keyframe by blending
            # the last N frames toward the known target keyframe image.
            if lock_end_keyframe and out_list:
                target = end_img.resize(out_list[0].size).convert("RGB")
                blend_n = max(1, min(end_blend_frames, len(out_list)))
                for j in range(1, blend_n + 1):
                    idx = len(out_list) - blend_n + (j - 1)
                    alpha = j / float(blend_n)
                    out_list[idx] = self.PILImage.blend(out_list[idx].convert("RGB"), target, alpha)

            tmp_seg = f"/tmp/hunyuan_actor_seg_{i + 1:03d}.mp4"
            export_to_video(out_list, tmp_seg, fps=fps)
            dst = out_base / f"segment_{i + 1:03d}.mp4"
            dst.write_bytes(Path(tmp_seg).read_bytes())
            segment_paths.append(str(dst))

        all_frames = []
        for p in segment_paths:
            all_frames.append(iio.imread(p, index=None))
        combined = self.np.concatenate(all_frames, axis=0)
        final_mp4 = out_base / "concatenated.mp4"
        iio.imwrite(final_mp4, combined, fps=fps)

        meta = {
            "video_run_id": video_run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "keyframe_run_id": keyframe_run_id,
            "num_keyframes": len(keyframes),
            "num_segments": len(segment_paths),
            "config": {
                "i2v_model": HUNYUAN_I2V_MODEL_ID,
                "gpu": "H100",
                "num_frames_per_segment": num_frames,
                "i2v_steps": i2v_steps,
                "fps": fps,
                "lock_end_keyframe": lock_end_keyframe,
                "end_blend_frames": end_blend_frames,
            },
            "timing": {
                "cold_start_s": self.cold_start_s,
                "per_segment_s": segment_timings,
                "inference_total_s": round(sum(segment_timings), 2),
                "avg_segment_s": round(sum(segment_timings) / max(len(segment_timings), 1), 2),
            },
            "transition_prompts": transition_prompts,
            "segment_paths": segment_paths,
            "concatenated_path": str(final_mp4),
        }
        (out_base / "video_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output_volume.commit()
        return meta


@app.local_entrypoint()
def main(
    keyframe_run_id: str,
    scenario_path: str = "",
    lock_end_keyframe: bool = True,
    end_blend_frames: int = 6,
    i2v_steps: int = DEFAULT_I2V_STEPS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    fps: int = DEFAULT_FPS,
):
    transition_prompts = None
    if scenario_path:
        transition_prompts = _load_transition_prompts_from_scenario(Path(scenario_path))

    video_run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S") + "_actor_hunyuan"
    print(f"Creating Hunyuan video from keyframes: {keyframe_run_id}")
    print(f"Video run ID: {video_run_id}")
    print(f"Output: my-volume/{VIDEO_SUBFOLDER}/{video_run_id}/")
    if scenario_path:
        print(f"Scenario: {scenario_path}")
    print(
        f"Config: i2v_steps={i2v_steps}, num_frames={num_frames}, fps={fps}, "
        f"lock_end_keyframe={lock_end_keyframe}, end_blend_frames={end_blend_frames}"
    )

    meta = HunyuanFromKeyframes().run.remote(
        keyframe_run_id=keyframe_run_id,
        video_run_id=video_run_id,
        scenario_transition_prompts=transition_prompts,
        lock_end_keyframe=lock_end_keyframe,
        end_blend_frames=end_blend_frames,
        i2v_steps=i2v_steps,
        num_frames=num_frames,
        fps=fps,
    )

    print("Done.")
    print(json.dumps(meta, indent=2))
