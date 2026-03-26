"""
Modal app: 6-segment video chain (T2V -> I2V x5).
Segment 1: text-to-video. Segments 2–6: image-to-video from the previous segment's last frame.
Each segment ~5–6 s. Outputs: segment_001–006.mp4, concatenated.mp4, chain_meta.json in my-volume/video_chain/<run_id>/.

Optimised for H100 (80 GB VRAM): no CPU offloading, reduced I2V steps.

Usage:
  modal run modal_video_compare.py::app.main_chain
  modal run modal_video_compare.py::app.main_chain --input-path chain_prompts.json
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = "/cache"
MINUTES = 60
OUTPUT_DIR = "/vol/out"
NUM_SEGMENTS = 6

HUNYUAN_FAST_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
HUNYUAN_FAST_I2V_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
VIDEO_CHAIN_SUBFOLDER = "video_chain"

# ~5–6 s per segment @ 15 fps
HUNYUAN_FAST_FRAMES = 81   # ~5.4 s
HUNYUAN_FAST_STEPS = 12    # T2V inference steps
HUNYUAN_FAST_I2V_STEPS = 20 # I2V inference steps (was 50, reduced for speed)

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=0.33.0",
        "diffusers>=0.36.0",
        "ftfy",
        "imageio[ffmpeg]",
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
app = modal.App("bloodcode-video-chain")


def get_last_frame(frames: list) -> "PIL.Image.Image":
    """Extract the last frame from a list of video frames (PIL or numpy). Returns PIL.Image."""
    import PIL.Image
    import numpy as np
    last = frames[-1]
    if isinstance(last, PIL.Image.Image):
        return last
    if isinstance(last, np.ndarray):
        if last.max() <= 1.0:
            last = (last * 255).astype(np.uint8)
        return PIL.Image.fromarray(last)
    raise TypeError(f"Unsupported frame type: {type(last)}")


def _frames_to_pil_list(out, np_module, PILImage):
    if isinstance(out, np_module.ndarray):
        return [PILImage.fromarray((f * 255).astype("uint8")) for f in out]
    return list(out)


# ---------------------------------------------------------------------------
# Chain runner: T2V segment 1 -> I2V segments 2..N from last frame each time
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="H100",
    timeout=60 * MINUTES,
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
)
class HunyuanChainRunner:
    """Runs an N-segment chain: T2V (segment 1) then I2V (segments 2..N from previous last frame)."""

    @modal.enter()
    def load(self):
        from diffusers import HunyuanVideo15ImageToVideoPipeline, HunyuanVideo15Pipeline
        import PIL.Image as PILImage
        import numpy as np

        t0 = time.time()

        self.pipe_t2v = HunyuanVideo15Pipeline.from_pretrained(
            HUNYUAN_FAST_MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe_t2v.vae.enable_tiling()

        self.pipe_i2v = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
            HUNYUAN_FAST_I2V_MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe_i2v.vae.enable_tiling()

        self.PILImage = PILImage
        self.np = np
        self.cold_start_s = round(time.time() - t0, 2)

    @modal.method()
    def run_chain(
        self,
        prompts: list[str],
        volume_subdir: str,
        num_frames: int = HUNYUAN_FAST_FRAMES,
        num_inference_steps_t2v: int = HUNYUAN_FAST_STEPS,
        num_inference_steps_i2v: int = HUNYUAN_FAST_I2V_STEPS,
        seed: int | None = None,
        fps: int = 15,
    ) -> dict:
        import imageio.v3 as iio

        n = len(prompts)
        if n < 2:
            raise ValueError("prompts must contain at least 2 segments")
        if n > 20:
            raise ValueError("prompts limited to 20 segments")

        if seed is not None:
            torch.manual_seed(seed)

        base = Path(OUTPUT_DIR) / VIDEO_CHAIN_SUBFOLDER / volume_subdir
        base.mkdir(parents=True, exist_ok=True)

        segment_paths = []
        segment_timings = []
        last_frame = None

        t_total = time.time()

        for i, prompt in enumerate(prompts):
            seg_num = i + 1
            seg_name = f"segment_{seg_num:03d}.mp4"
            tmp_path = f"/tmp/chain_seg{seg_num}.mp4"

            t_seg = time.time()
            if i == 0:
                out = self.pipe_t2v(
                    prompt=prompt,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps_t2v,
                    output_type="pil",
                ).frames[0]
            else:
                out = self.pipe_i2v(
                    prompt=prompt,
                    image=last_frame,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps_i2v,
                    output_type="pil",
                ).frames[0]
            dt_seg = round(time.time() - t_seg, 2)
            segment_timings.append(dt_seg)
            print(f"  Segment {seg_num}/{n}: {dt_seg:.1f}s", flush=True)

            out_list = _frames_to_pil_list(out, self.np, self.PILImage)
            export_to_video(out_list, tmp_path, fps=fps)
            (base / seg_name).write_bytes(Path(tmp_path).read_bytes())
            segment_paths.append(tmp_path)
            last_frame = get_last_frame(out_list)

        all_frames = []
        for p in segment_paths:
            all_frames.append(iio.imread(p, index=None))
        combined = self.np.concatenate(all_frames, axis=0)
        iio.imwrite(base / "concatenated.mp4", combined, fps=fps)

        dt_total = round(time.time() - t_total, 2)
        inference_total = round(sum(segment_timings), 2)

        meta = {
            "gpu": "H100",
            "num_frames_per_segment": num_frames,
            "t2v_steps": num_inference_steps_t2v,
            "i2v_steps": num_inference_steps_i2v,
            "fps": fps,
            "seed": seed,
            "timing": {
                "cold_start_s": self.cold_start_s,
                "inference_total_s": inference_total,
                "generation_wall_s": dt_total,
                "per_segment_s": segment_timings,
            },
        }
        for i, p in enumerate(prompts):
            meta[f"prompt_{i + 1}"] = p
        (base / "chain_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output_volume.commit()

        result = {
            "volume_subdir": volume_subdir,
            "segment_paths": [str(base / f"segment_{j+1:03d}.mp4") for j in range(n)],
            "concatenated": str(base / "concatenated.mp4"),
            "cold_start_s": self.cold_start_s,
            "segment_timings": segment_timings,
            "inference_total_s": inference_total,
            "generation_wall_s": dt_total,
        }
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "A cat walking in the rain, cinematic, soft focus.",
    "The same cat jumps up and runs forward excitedly through the rain, dynamic motion.",
    "The same cat lies down on the ground, relaxing, cinematic.",
    "The same cat notices a mouse and starts chasing it, excited, dynamic motion.",
    "The same cat starts to dance, playful, fun movement.",
    "The same cat continues to dance with a slight variation, playful until the end, cinematic.",
]


@app.local_entrypoint()
def main_chain(
    prompt_1: str = "",
    prompt_2: str = "",
    prompt_3: str = "",
    prompt_4: str = "",
    prompt_5: str = "",
    prompt_6: str = "",
    seed: int | None = None,
    input_path: str = "",
):
    """Run 6-segment chain (T2V -> I2V x5). Saves segment_001–006.mp4, concatenated.mp4, chain_meta.json to my-volume/video_chain/<run_id>/."""
    prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5, prompt_6]
    if input_path.strip() and Path(input_path).exists():
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        prompts = [
            data.get(f"prompt_{i}", DEFAULT_PROMPTS[i - 1])
            for i in range(1, NUM_SEGMENTS + 1)
        ]
        seed = data.get("seed", seed)
    else:
        for i in range(NUM_SEGMENTS):
            if not (prompts[i] or "").strip():
                prompts[i] = DEFAULT_PROMPTS[i]
        prompts = prompts[:NUM_SEGMENTS]

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    print(f"\n{'=' * 60}")
    print(f"  HunyuanVideo Chain – H100, no offloading, {HUNYUAN_FAST_I2V_STEPS} I2V steps")
    print(f"{'=' * 60}")
    print(f"  Segments : {NUM_SEGMENTS}  (T2V x1 + I2V x{NUM_SEGMENTS - 1})")
    print(f"  T2V steps: {HUNYUAN_FAST_STEPS}   |  I2V steps: {HUNYUAN_FAST_I2V_STEPS}")
    print(f"  Frames   : {HUNYUAN_FAST_FRAMES} per segment  (~{HUNYUAN_FAST_FRAMES / 15:.1f}s @ 15fps)")
    print(f"  Seed     : {seed}")
    print(f"  Run ID   : {run_id}")
    print(f"  Output   : my-volume/{VIDEO_CHAIN_SUBFOLDER}/{run_id}/")
    print(f"{'=' * 60}")
    for i, p in enumerate(prompts):
        print(f"  Segment {i + 1}: {p[:60]}...")
    print(f"{'=' * 60}\n")

    t_wall = time.time()
    result = HunyuanChainRunner().run_chain.remote(
        prompts=prompts,
        volume_subdir=run_id,
        seed=seed,
    )
    dt_wall = time.time() - t_wall

    print(f"\n{'=' * 60}")
    print("  COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Cold start       : {result['cold_start_s']:.1f}s")
    print(f"  Inference total  : {result['inference_total_s']:.1f}s")
    for i, t in enumerate(result["segment_timings"]):
        label = "T2V" if i == 0 else "I2V"
        print(f"    Segment {i + 1} ({label}) : {t:.1f}s")
    print(f"  Generation wall  : {result['generation_wall_s']:.1f}s")
    print(f"  Total wall time  : {dt_wall:.1f}s  (incl. cold start + network)")
    print(f"{'=' * 60}")
    for path in result["segment_paths"]:
        print(f"  {Path(path).name} -> {path}")
    print(f"  concatenated.mp4 -> {result['concatenated']}")
    print(f"{'=' * 60}\n")
