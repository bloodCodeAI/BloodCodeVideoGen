# Parallel Diffusion Video Pipeline

## Overview

This pipeline generates multi-segment video by splitting the work across multiple
H100 GPUs running simultaneously, rather than processing segments sequentially on
a single GPU.

**Key difference from `modal_video_compare.py` (sequential chain):**

| Aspect | Sequential chain | Parallel pipeline |
|---|---|---|
| Segment dependency | Each segment depends on previous segment's last frame | All segments are independent |
| GPU usage | 1x A100 for the entire duration | 1x H100 (keyframes) + Nx H100 (segments) |
| I2V steps | 50 | 20 (H100 compensates with faster compute) |
| Total wall time | N * segment_time (~3-5 min/seg) | max(segment_time) (~30-60s) |
| Continuity | Frame-to-frame continuous | Scene-to-scene (keyframe anchored) |

---

## Architecture

```
                       +-------------------+
                       |  Local entrypoint |
                       |   (orchestrator)  |
                       +---------+---------+
                                 |
              Phase 1a           |
         ┌───────────────────────┤
         |                       |
  +------v--------+             |
  | KeyframeGen   |             |
  | (1x H100)     |             |
  |               |             |
  | 1a: SDXL T2I  |             |       "reference.png"
  |  -> reference |             |       locks identity,
  |     image     |             |       colours, scene
  |               |             |
  | 1b: SDXL I2I  |             |       each keyframe
  |  -> N kframes |             |       starts from
  |  (from ref)   |             |       the reference
  +------+--------+             |
         |                       |
         | 6 PNG keyframes       |
         |                       |
              Phase 2            |
         ┌───────────────────────┤
         |   .spawn() x 6       |
         |                       |
  +------v--------+  +----------v---+        +----------+
  | SegmentGen #1 |  | SegmentGen #2| ...... | SegGen #6|
  | (H100)        |  | (H100)       |        | (H100)   |
  | HunyuanVideo  |  | HunyuanVideo |        | HunyVid  |
  | I2V, 20 steps |  | I2V, 20 steps|        | I2V 20st |
  +------+--------+  +------+-------+        +----+-----+
         |                   |                     |
         | segment_001.mp4   | segment_002.mp4     | segment_006.mp4
         |                   |                     |
         └──────────────┬────┴─────────────────────┘
                        |
              Phase 3   |
                 +------v------+
                 |  Assembler  |
                 |  (CPU)      |
                 |  concat +   |
                 |  metrics    |
                 +------+------+
                        |
                        v
                 concatenated.mp4
                 metrics.json
```

---

## Pipeline Stages

### Phase 1a – Reference Image

- **Model:** Stable Diffusion XL (text-to-image)
- **Hardware:** 1x H100 SXM
- **Output:** `reference.png` – one hero image that establishes the character
- **Time:** ~2-4s

The reference prompt describes the character in a neutral pose with specific physical
details (fur colour, eye colour, markings).  This image becomes the pixel-level
anchor for all keyframes, ensuring consistent identity.

### Phase 1b – Keyframe Generation (img2img)

- **Model:** Stable Diffusion XL (image-to-image from the reference)
- **Hardware:** Same GPU as Phase 1a (no extra container)
- **Outputs:** 6 PNG images, each derived from the reference but showing a new pose
- **Time:** ~1-3s per image (faster than T2I due to `strength < 1.0`)
- **`strength` parameter:** Controls how much each keyframe deviates from the
  reference.  Default 0.6 = substantial pose changes while keeping the cat's
  appearance.  Lower (0.4) = more faithful.  Higher (0.75) = more creative.

Because every keyframe starts from the same reference pixels, the cat's fur pattern,
colouring, eye colour, and environment lighting are inherited automatically.

### Phase 2 – Segment Generation

- **Model:** HunyuanVideo 1.5 (480p, Image-to-Video)
- **Hardware:** 6x H100 SXM (one container per segment, spawned in parallel)
- **Inference steps:** 20 (reduced from 50 in the sequential pipeline)
- **Frames per segment:** 81 (~5.4s at 15fps)
- **Outputs:** 6 MP4 files + the keyframe PNG saved alongside each

Each segment receives its keyframe image + a motion-oriented prompt and generates
video independently.  All 6 segments run simultaneously on separate GPUs.

### Phase 3 – Assembly

- **Hardware:** CPU only (no GPU needed)
- **Process:** Reads all segment MP4s from the volume, concatenates frames with
  numpy, writes `concatenated.mp4` and `metrics.json`

---

## Optimizations Applied

| Optimization | Detail | Impact |
|---|---|---|
| **H100 SXM GPU** | ~2x faster BF16 compute vs A100, higher memory bandwidth | ~2x inference speedup per segment |
| **BF16 precision** | Native bf16 on H100 tensor cores | Full precision, no quality loss |
| **Reduced I2V steps** | 20 steps instead of 50 | ~2.5x faster per segment |
| **Full parallelism** | N segments on N GPUs simultaneously | Wall time = slowest segment, not sum |
| **Separate container images** | SDXL image is lighter than I2V image | Faster cold starts for keyframe phase |
| **VAE tiling** | Processes VAE in tiles to reduce peak VRAM | Prevents OOM at 480p with 81 frames |
| **CPU offloading** | Model components move to CPU when idle | Fits within H100 80GB VRAM budget |
| **Reference img2img** | One hero image anchors all keyframes via SDXL I2I | Consistent character identity without LoRA/IP-Adapter |

### Not yet applied (future improvements)

| Optimization | Why not yet | Expected gain |
|---|---|---|
| **FP8 quantization** | Requires `optimum-quanto` + model compatibility testing | ~1.5x additional speedup on H100 |
| **TensorRT compilation** | One-time ~60-120s compile cost doesn't amortize for single-shot | ~2x for repeated inference |
| **torch.compile** | Same amortization issue for single-shot | ~1.3-1.5x for repeated inference |
| **DynamiCrafter interpolation** | Not in diffusers; requires custom codebase integration | Conditions on both start+end frames |
| **IP-Adapter consistency** | Reference img2img approach already handles most of this | Even stricter identity across keyframes |

---

## Output Structure

```
my-volume/parallel_interp/<run_id>/
  reference.png           # Hero image: character identity anchor
  keyframe_001.png        # img2img from reference – segment 1 starting point
  keyframe_002.png
  ...
  keyframe_006.png
  segment_001.mp4         # HunyuanVideo I2V output from keyframe 1
  segment_002.mp4
  ...
  segment_006.mp4
  concatenated.mp4        # All 6 segments joined back-to-back
  metrics.json            # Full timing and configuration data
```

---

## Metrics (metrics.json)

The `metrics.json` file saved alongside the video contains:

### Configuration
- `gpu` – GPU type used (H100)
- `num_segments` – number of parallel segments
- `frames_per_segment` / `total_frames` – frame counts
- `i2v_inference_steps` / `sdxl_inference_steps` – denoising steps
- `kf_strength` – img2img strength used to derive keyframes from reference
- `fps` / `resolution` – output format

### Timing
- `pipeline_total_s` – end-to-end wall time including all overhead
- `phase1_total_s` – Phase 1 wall time (reference + all keyframes, includes cold start)
- `phase1_reference_s` – time for the reference image alone
- `phase1_per_keyframe_s` – time for each individual keyframe (img2img)
- `phase2_segments_wall_s` – Phase 2 wall time (all segments in parallel)
- `phase2_per_segment_s` – per-segment inference time (on-GPU only)
- `phase2_slowest_s` / `phase2_fastest_s` / `phase2_avg_s` – segment stats
- `phase3_assembly_s` – concatenation time
- `sequential_estimate_s` – estimated time if segments ran sequentially
- `parallel_speedup_x` – `sequential_estimate / pipeline_total`

### Output
- `video_duration_s` – total video length in seconds
- `concatenated_path` / `segment_paths` – file paths on the volume

### Prompts
- `reference` – the prompt used to generate the hero image
- All keyframe and segment prompts stored for reproducibility

---

## Usage

### Basic run (default cat prompts)

```bash
cd BloodCodeVideoGen
modal run modal_parallel_interp.py
```

### With a fixed seed

```bash
modal run modal_parallel_interp.py --seed 42
```

### Adjust keyframe variation (img2img strength)

```bash
modal run modal_parallel_interp.py --kf-strength 0.5   # closer to reference
modal run modal_parallel_interp.py --kf-strength 0.75  # more creative poses
```

### With custom prompts (JSON)

```bash
modal run modal_parallel_interp.py --input-path parallel_prompts.json
```

The JSON supports `reference_prompt`, `kf_strength`, per-segment keyframe/segment
prompts, and `seed`.  See `parallel_prompts_example.json` for the expected format.

---

## Trade-offs and Known Limitations

1. **Character consistency (largely solved by the reference image).**
   All keyframes are derived via img2img from a single reference, so the cat's fur
   pattern, eye colour, and overall appearance are inherited.  At higher `strength`
   values (> 0.7) the model has more freedom and identity can drift.  Tune `strength`
   down to 0.4–0.5 if you need tighter consistency.  For even stricter control,
   add a character LoRA or IP-Adapter.

2. **No end-frame conditioning.**
   HunyuanVideo I2V only conditions on the starting keyframe, not the ending one.
   Segments don't smoothly transition into the next segment's keyframe.  Mitigation:
   swap in DynamiCrafter (conditions on both start and end frames) when integrated,
   or apply cross-fade blending at segment boundaries.

3. **Cold start overhead.**
   First run spins up N+1 GPU containers and downloads models.  Subsequent runs with
   warm containers are significantly faster.  Use `keep_warm=N` on the class
   decorators for production workloads.

4. **Step reduction quality.**
   20 I2V steps (vs 50 in the sequential pipeline) may produce slightly less detailed
   motion.  Test with your specific content; bump to 30 if quality is insufficient.

5. **Strength tuning.**
   `kf_strength` is the single most impactful dial for visual quality vs. pose
   variety.  Run a quick test with 2-3 values (0.4, 0.6, 0.75) to find the sweet
   spot for your content before committing to a full pipeline run.
