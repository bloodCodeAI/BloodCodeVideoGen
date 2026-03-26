from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import modal

CACHE_DIR = "/cache"
OUTPUT_DIR = "/vol/out"
WAN_REPO_DIR = "/opt/Wan2.1"
MINUTES = 60

KEYFRAME_SUBFOLDER = "actor_pipeline/keyframes"
VIDEO_SUBFOLDER = "actor_pipeline/two_endpoint_video"

DEFAULT_NUM_FRAMES = 81
DEFAULT_FPS = 15
DEFAULT_BACKEND = "wan_native_cli"
DEFAULT_WAN_CKPT_DIR = "/vol/out/models/Wan2.1-FLF2V-14B-720P"
DEFAULT_WAN_SIZE = "1280*720"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "build-essential")
    .uv_pip_install(
        "accelerate>=1.1.1",
        "dashscope",
        "diffusers>=0.31.0",
        "easydict",
        "einops",
        "ftfy",
        "imageio[ffmpeg]",
        "numpy",
        "opencv-python>=4.9.0.80",
        "packaging",
        "Pillow",
        "psutil",
        "requests",
        "safetensors",
        "sentencepiece",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.49.0",
    )
    .run_commands(
        f"git clone https://github.com/Wan-Video/Wan2.1.git {WAN_REPO_DIR}",
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install ninja packaging",
        "python -m pip install flash-attn --no-build-isolation",
    )
    .env(
        {
            "HF_HUB_CACHE": CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "8",
        }
    )
)

cache_volume = modal.Volume.from_name("video-gen-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("my-volume")
app = modal.App("actor-two-endpoint-from-keyframes")


@dataclass
class SegmentSpec:
    idx: int
    start_keyframe: Path
    end_keyframe: Path
    transition_prompt: str


def _load_transition_prompts_from_scenario(scenario_path: Path) -> list[str]:
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {scenario_path}")
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("Scenario JSON must include at least 2 steps.")
    sorted_steps = sorted([s for s in steps if isinstance(s, dict)], key=lambda x: x.get("k", 0))
    style = data.get("global_style")
    style_suffix = f" {style}" if isinstance(style, str) and style.strip() else ""
    prompts: list[str] = []
    for i in range(len(sorted_steps) - 1):
        a = str(sorted_steps[i].get("prompt", "")).strip()
        b = str(sorted_steps[i + 1].get("prompt", "")).strip()
        if not a or not b:
            raise ValueError(f"Transition {i + 1}->{i + 2} missing non-empty prompt.")
        prompts.append(
            "Smoothly transition between the provided first and last frame. "
            f"Start: {a}. End: {b}.{style_suffix}"
        )
    return prompts


def _make_segment_specs(keyframes: list[Path], transition_prompts: list[str] | None) -> list[SegmentSpec]:
    if len(keyframes) < 2:
        raise ValueError(f"Need at least 2 keyframes, found {len(keyframes)}")
    if transition_prompts:
        needed = len(keyframes) - 1
        if len(transition_prompts) < needed:
            raise ValueError(
                f"Scenario transitions ({len(transition_prompts)}) are fewer than keyframe pairs ({needed})."
            )
        if len(transition_prompts) > needed:
            print(
                f"Truncating scenario transitions from {len(transition_prompts)} to {needed} for selected keyframes.",
                flush=True,
            )
            transition_prompts = transition_prompts[:needed]
    specs: list[SegmentSpec] = []
    for i in range(len(keyframes) - 1):
        prompt = (
            transition_prompts[i]
            if transition_prompts
            else "Generate a realistic, smooth transition from the first frame to the last frame."
        )
        specs.append(
            SegmentSpec(
                idx=i + 1,
                start_keyframe=keyframes[i],
                end_keyframe=keyframes[i + 1],
                transition_prompt=prompt,
            )
        )
    return specs


def _generate_segment_with_backend(
    *,
    backend: str,
    spec: SegmentSpec,
    num_frames: int,
    out_path: Path,
    wan_ckpt_dir: str,
    wan_size: str,
    wan_use_prompt_extend: bool,
) -> None:
    if backend != "wan_native_cli":
        raise ValueError(f"Unsupported backend: {backend}")

    if not Path(wan_ckpt_dir).exists():
        raise FileNotFoundError(f"Wan checkpoint directory not found: {wan_ckpt_dir}")

    from PIL import Image
    size_map = {
        "1280*720": (1280, 720),
        "720*1280": (720, 1280),
        "832*480": (832, 480),
        "480*832": (480, 832),
    }
    if wan_size not in size_map:
        raise ValueError(
            f"Unsupported wan_size '{wan_size}'. Supported: {', '.join(size_map.keys())}"
        )
    target_size = size_map[wan_size]

    tmp_dir = Path("/tmp") / f"wan_pair_{spec.idx:03d}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    first_frame = tmp_dir / "first_frame.png"
    last_frame = tmp_dir / "last_frame.png"
    rendered_mp4 = tmp_dir / "output.mp4"

    # Wan FLF2V supports only a fixed set of sizes; always normalize both endpoints.
    Image.open(spec.start_keyframe).convert("RGB").resize(target_size, Image.Resampling.LANCZOS).save(first_frame)
    Image.open(spec.end_keyframe).convert("RGB").resize(target_size, Image.Resampling.LANCZOS).save(last_frame)

    cmd = [
        "python",
        f"{WAN_REPO_DIR}/generate.py",
        "--task",
        "flf2v-14B",
        "--size",
        wan_size,
        "--ckpt_dir",
        wan_ckpt_dir,
        "--first_frame",
        str(first_frame),
        "--last_frame",
        str(last_frame),
        "--prompt",
        spec.transition_prompt,
        "--save_file",
        str(rendered_mp4),
        "--frame_num",
        str(num_frames),
    ]
    if wan_use_prompt_extend:
        cmd.append("--use_prompt_extend")

    env = os.environ.copy()
    env["HF_HOME"] = CACHE_DIR
    env["HF_HUB_CACHE"] = CACHE_DIR
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Wan CLI failed for segment {spec.idx}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-5000:]}\n"
            f"STDERR:\n{proc.stderr[-5000:]}"
        )
    if not rendered_mp4.exists():
        raise FileNotFoundError(f"Wan CLI finished but output video not found: {rendered_mp4}")
    out_path.write_bytes(rendered_mp4.read_bytes())


def _concat_segments(segment_paths: list[Path], out_path: Path, fps: int) -> None:
    import imageio.v3 as iio
    import numpy as np

    all_frames = []
    for i, p in enumerate(segment_paths):
        seg_frames = iio.imread(p, index=None)
        if i > 0 and len(seg_frames) > 0:
            seg_frames = seg_frames[1:]
        all_frames.append(seg_frames)
    combined = np.concatenate(all_frames, axis=0)
    iio.imwrite(out_path, combined, fps=fps)


@app.function(
    image=image,
    gpu="H100",
    timeout=180 * MINUTES,
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
)
def run_two_endpoint(
    keyframe_run_id: str,
    video_run_id: str,
    scenario_path: str = "",
    scenario_transition_prompts: list[str] | None = None,
    backend: str = DEFAULT_BACKEND,
    num_frames: int = DEFAULT_NUM_FRAMES,
    fps: int = DEFAULT_FPS,
    max_keyframes: int = 0,
    wan_ckpt_dir: str = DEFAULT_WAN_CKPT_DIR,
    wan_size: str = DEFAULT_WAN_SIZE,
    wan_use_prompt_extend: bool = False,
) -> dict:
    t0 = time.time()
    output_volume.reload()

    keyframe_base = Path(OUTPUT_DIR) / KEYFRAME_SUBFOLDER / keyframe_run_id
    keyframe_dir = keyframe_base / "keyframes"
    if not keyframe_dir.exists():
        raise FileNotFoundError(f"Keyframe folder not found: {keyframe_dir}")
    keyframes = sorted(keyframe_dir.glob("keyframe_*.png"))
    if max_keyframes and max_keyframes > 1:
        keyframes = keyframes[:max_keyframes]
        print(f"Using first {len(keyframes)} keyframes for smoke test", flush=True)

    transition_prompts = scenario_transition_prompts
    specs = _make_segment_specs(keyframes, transition_prompts)

    out_base = Path(OUTPUT_DIR) / VIDEO_SUBFOLDER / video_run_id
    out_base.mkdir(parents=True, exist_ok=True)

    segment_paths: list[Path] = []
    segment_timings: list[float] = []

    for spec in specs:
        print(f"[{spec.idx}/{len(specs)}] {spec.start_keyframe.name} -> {spec.end_keyframe.name}", flush=True)
        print(f"[{spec.idx}/{len(specs)}] Prompt: {spec.transition_prompt}", flush=True)
        dst = out_base / f"segment_{spec.idx:03d}.mp4"
        t_seg = time.time()
        _generate_segment_with_backend(
            backend=backend,
            spec=spec,
            num_frames=num_frames,
            out_path=dst,
            wan_ckpt_dir=wan_ckpt_dir,
            wan_size=wan_size,
            wan_use_prompt_extend=wan_use_prompt_extend,
        )
        dt = round(time.time() - t_seg, 2)
        segment_timings.append(dt)
        segment_paths.append(dst)
        print(f"[{spec.idx}/{len(specs)}] Segment done in {dt:.2f}s", flush=True)

    final_mp4 = out_base / "concatenated.mp4"
    _concat_segments(segment_paths, final_mp4, fps=fps)

    meta = {
        "video_run_id": video_run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "keyframe_run_id": keyframe_run_id,
        "scenario_path": scenario_path,
        "backend": backend,
        "num_keyframes": len(keyframes),
        "num_segments": len(segment_paths),
        "config": {
            "num_frames_per_segment": num_frames,
            "fps": fps,
            "wan_ckpt_dir": wan_ckpt_dir,
            "wan_size": wan_size,
            "wan_use_prompt_extend": wan_use_prompt_extend,
            "max_keyframes": max_keyframes,
            "endpoint_conditioning": "first+last",
        },
        "timing": {
            "per_segment_s": segment_timings,
            "inference_total_s": round(sum(segment_timings), 2),
            "pipeline_total_s": round(time.time() - t0, 2),
        },
        "segment_paths": [str(p) for p in segment_paths],
        "concatenated_path": str(final_mp4),
    }
    (out_base / "video_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    output_volume.commit()
    return meta


@app.local_entrypoint()
def main(
    keyframe_run_id: str,
    scenario_path: str = "",
    backend: str = DEFAULT_BACKEND,
    num_frames: int = DEFAULT_NUM_FRAMES,
    fps: int = DEFAULT_FPS,
    max_keyframes: int = 0,
    wan_ckpt_dir: str = DEFAULT_WAN_CKPT_DIR,
    wan_size: str = DEFAULT_WAN_SIZE,
    wan_use_prompt_extend: bool = False,
):
    scenario_transition_prompts = None
    if scenario_path:
        scenario_transition_prompts = _load_transition_prompts_from_scenario(Path(scenario_path))

    video_run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S") + "_two_endpoint"
    print(f"Creating two-endpoint video from keyframes: {keyframe_run_id}")
    print(f"Video run ID: {video_run_id}")
    print(f"Output: my-volume/{VIDEO_SUBFOLDER}/{video_run_id}/")
    print(f"Backend: {backend}")
    if scenario_path:
        print(f"Scenario: {scenario_path}")
    print(
        f"Config: num_frames={num_frames}, fps={fps}, max_keyframes={max_keyframes}, "
        f"wan_size={wan_size}, wan_ckpt_dir={wan_ckpt_dir}"
    )

    meta = run_two_endpoint.remote(
        keyframe_run_id=keyframe_run_id,
        video_run_id=video_run_id,
        scenario_path=scenario_path,
        scenario_transition_prompts=scenario_transition_prompts,
        backend=backend,
        num_frames=num_frames,
        fps=fps,
        max_keyframes=max_keyframes,
        wan_ckpt_dir=wan_ckpt_dir,
        wan_size=wan_size,
        wan_use_prompt_extend=wan_use_prompt_extend,
    )
    print("Done.")
    print(json.dumps(meta, indent=2))
