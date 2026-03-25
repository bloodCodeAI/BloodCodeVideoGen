# Hybrid Pipeline: Diffusion Keyframes + RIFE Interpolation

## Overview

This pipeline generates ~37 seconds of video by combining two models:

1. **SDXL** (diffusion) generates 36 keyframe images from a single reference
2. **RIFE** (frame interpolation) fills in smooth motion between each keyframe pair

This is dramatically cheaper and faster than the all-diffusion parallel pipeline
because RIFE runs on a single T4 GPU and processes each frame in milliseconds.

**Comparison with the all-diffusion approach (`modal_parallel_interp.py`):**

| Aspect | All-diffusion | Hybrid (this pipeline) |
|---|---|---|
| Phase 2 model | HunyuanVideo I2V (diffusion) | RIFE v4.22 (interpolation) |
| Phase 2 GPU | 6x H100 | 1x T4 |
| Phase 2 cost | ~$3-6 | ~$0.01 |
| Phase 2 wall time | ~60s | ~20s |
| Motion type | Semantically rich actions | Smooth pixel interpolation |
| Video length | ~32s (6 segments x 5.4s) | ~37s (561 frames / 15fps) |
| Character consistency | Good (reference img2img) | Good (same approach + interpolation is pixel-perfect) |

---

## Architecture

```
Phase 1 (1x H100, SDXL)
  reference.png ──> keyframe_001.png (I2I)
                ──> keyframe_002.png (I2I)
                ──> ...
                ──> keyframe_036.png (I2I)

Phase 2 (1x T4, RIFE v4.22)
  KF01 + KF02 ──> 15 interpolated frames
  KF02 + KF03 ──> 15 interpolated frames
  ...
  KF35 + KF36 ──> 15 interpolated frames

Phase 3 (CPU)
  561 frames ──> concatenated.mp4 + metrics.json
```

## Frame Math

- 36 keyframes, 35 gaps between adjacent pairs
- RIFE exp=4: each gap gets 2^4 - 1 = 15 interpolated frames
- Total unique frames: 35 x 15 + 36 = **561**
- At 15 fps: **37.4 seconds** of video

Adjustable via `NUM_KEYFRAMES` and `RIFE_EXP` in the config.

---

## How RIFE Works

RIFE (Real-Time Intermediate Flow Estimation) takes two images and predicts what
the scene looks like at any point between them:

```
model.inference(frame_A, frame_B, ratio=0.5)  -->  midpoint frame
model.inference(frame_A, frame_B, ratio=0.25) -->  quarter-way frame
```

For exp=4, we generate 15 intermediate frames at evenly spaced ratios
(1/16, 2/16, ..., 15/16) between each keyframe pair.

**RIFE v4.22** is specifically optimised for diffusion-model-generated content,
which is ideal since our keyframes come from SDXL.

Key characteristics:
- ~50MB model (vs ~25GB for HunyuanVideo)
- ~1-2 GB VRAM (runs on T4, L4, or even CPU)
- ~0.5s per pair at 848x480 on T4
- Zero hallucination: output is purely derived from the input pixels
- Works best when adjacent keyframes are visually similar

---

## Prompt Design Strategy

Since RIFE interpolates **pixels**, adjacent keyframes need to be visually similar.
The prompts are structured as 6 story beats with 6 gradual variations each:

| Keyframes | Beat | Motion gradient |
|---|---|---|
| 1-6 | Walking in rain | left paw, right paw, look left, look right... |
| 7-12 | Speeding up | trotting, running, crouching, preparing to leap... |
| 13-18 | Jumping and landing | mid-leap, peak, descending, landing, standing... |
| 19-24 | Slowing to rest | walking slow, stopping, sitting, lying down... |
| 25-30 | Noticing mouse | ears perk, head up, crouch, stalk, pounce... |
| 31-36 | Dancing | stand on hind legs, dance, twirl, spin, final pose... |

Each prompt describes a small pose increment from the previous one, giving RIFE
minimal visual distance to interpolate across.

---

## Pipeline Stages in Detail

### Phase 1 -- Reference + Keyframes

- **Model:** SDXL (T2I for reference, I2I for keyframes)
- **Hardware:** 1x H100
- **Cold start:** Measured and logged (model download + GPU init)
- **Reference:** One hero image at full quality (25 SDXL steps)
- **Keyframes:** 36 images via img2img from the reference
  - `strength=0.6`: enough to change pose, keeps cat identity
  - Effective steps per keyframe: int(25 * 0.6) = 15, so faster than T2I

### Phase 2 -- RIFE Interpolation

- **Model:** RIFE v4.22 (Practical-RIFE repo, cloned into container)
- **Hardware:** 1x T4 (cheapest GPU on Modal, ~$0.27/hr)
- **Cold start:** Measured and logged (git clone happens at image build, model load ~2s)
- **Processing:** 35 pairs sequential, ~0.5s each
- **Padding:** Images padded to multiples of 64 for RIFE, then cropped back

### Phase 3 -- Assembly

- **Hardware:** CPU only
- **Process:** Receives all frames as PNG bytes, stacks into numpy array, writes MP4
- **Also saves:** metrics.json with all timing data

---

## Output Structure

```
my-volume/hybrid_interp/<run_id>/
  reference.png                    # Character identity anchor
  keyframe_001.png ... 036.png     # SDXL img2img outputs
  concatenated.mp4                 # Final video (561 frames, ~37s)
  metrics.json                     # Full timing and config data
```

---

## Metrics (metrics.json)

### Timing breakdown
- `phase1_wall_s` -- total Phase 1 including cold start
- `phase1_cold_start_s` -- SDXL model load time
- `phase1_inference_only_s` -- pure generation time (reference + keyframes)
- `phase2_wall_s` -- total Phase 2 including cold start
- `phase2_cold_start_s` -- RIFE model load time
- `phase2_inference_only_s` -- pure interpolation time
- `total_cold_start_s` -- sum of all cold starts
- `total_inference_only_s` -- sum of all inference (what you actually pay for)
- `pipeline_total_s` -- end-to-end wall time

### Cost awareness
The cold start metrics let you see exactly how much time (and cost) goes to
container spin-up vs. actual inference. On a budget:
- First run has full cold start (~30-120s depending on model cache)
- Subsequent runs within ~15 min reuse warm containers (cold start ~0)
- No `keep_warm` is set, so containers shut down immediately after use

---

## Usage

### Basic run

```bash
cd BloodCodeVideoGen
modal run modal_hybrid_interp.py
```

### Fixed seed + adjusted strength

```bash
modal run modal_hybrid_interp.py --seed 42 --kf-strength 0.5
```

### Custom prompts

```bash
modal run modal_hybrid_interp.py --input-path hybrid_prompts.json
```

---

## Tuning Guide

### `kf_strength` (most important)

Controls how much each keyframe deviates from the reference image:
- **0.4**: Very faithful to reference. Small pose changes. Best for RIFE quality.
- **0.6**: Default. Meaningful pose changes. Good RIFE results.
- **0.75+**: Large changes. RIFE may produce ghosting artifacts between very
  different keyframes.

For RIFE interpolation, **lower is generally better** because RIFE works best
when adjacent frames are visually close.

### `RIFE_EXP`

Controls frames per gap:
- **3**: 7 frames/gap. 36 kf -> 281 frames -> 18.7s. Faster, choppier.
- **4**: 15 frames/gap. 36 kf -> 561 frames -> 37.4s. Good default.
- **5**: 31 frames/gap. 36 kf -> 1121 frames -> 74.7s. Slower, may lose quality.

### `NUM_KEYFRAMES`

More keyframes = smoother result (less distance for RIFE to interpolate):
- 18 kf: ~18s video, cheaper, slightly rougher interpolation
- 36 kf: ~37s video, good balance (default)
- 72 kf: ~75s video, very smooth but ~2 min keyframe generation time

---

## Trade-offs and Limitations

1. **RIFE produces morphs, not actions.**
   Between "cat crouching" and "cat mid-leap", RIFE will create a smooth blend
   of the two poses. It won't generate the cat actually pushing off the ground.
   For semantically meaningful motion, use the all-diffusion pipeline instead.

2. **Adjacent keyframes must be visually similar.**
   The prompt design handles this by using small increments between adjacent
   keyframes. If you customize prompts, keep adjacent pairs close in content.

3. **Resolution constraint.**
   RIFE pads images to multiples of 64. At 848x480, this pads to 896x512
   internally, then crops back. No quality impact, but VRAM usage is slightly
   higher than the raw resolution suggests.

4. **No audio.**
   The pipeline outputs video only. For the fitness app use case, audio
   (exercise instructions) would need to be added in post-processing.
