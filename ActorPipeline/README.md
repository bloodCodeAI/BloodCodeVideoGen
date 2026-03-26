# Actor Pipeline (LoRA + IP-Adapter + ControlNet)

This folder contains a two-step pipeline:

1. Train an actor LoRA from photos in `BloodCodeVideoGen/ActorImages`
2. Generate consistent keyframes using:
   - trained LoRA (identity)
   - IP-Adapter (identity/style anchor)
   - ControlNet OpenPose (pose lock from pose images)

## Inputs

- Actor images: `BloodCodeVideoGen/ActorImages`
- Pose images: `BloodCodeVideoGen/posesImages/standardized_png_1024`

## Step 1 — Train LoRA

```bash
cd c:\Python\BloodCodeAi\BloodCodeVideoGen
modal run ActorPipeline/modal_train_actor_lora.py
```

Optional:

```bash
modal run ActorPipeline/modal_train_actor_lora.py --trigger-token sksactor --max-train-steps 1200 --learning-rate 1e-4
```

Output:

`my-volume/actor_pipeline/lora/<run_id>/`

Use this `<run_id>` in Step 2.

## Step 2 — Generate keyframes

```bash
modal run ActorPipeline/modal_generate_actor_keyframes.py --lora-run-id <run_id_from_step_1>
```

Optional tuning:

```bash
modal run ActorPipeline/modal_generate_actor_keyframes.py --lora-run-id <run_id> --ip-adapter-scale 0.65 --controlnet-scale 0.9 --num-inference-steps 30 --guidance-scale 6.0
```

Output:

`my-volume/actor_pipeline/keyframes/<run_id>/`

Contains:
- `actor_reference.png`
- `poses/pose_001.png ...`
- `keyframes/keyframe_001.png ...`
- `keyframe_meta.json`

## Notes

- `train_dreambooth_lora_sdxl.py` from Hugging Face diffusers is used for training.
- Training uses LoRA + 8-bit Adam for lower memory and faster optimization.
- For best identity quality, keep 20–30 diverse actor photos with full-body + face coverage.

## Two-endpoint transition skeleton

If you need strict `Ki -> K{i+1}` behavior (first+last frame conditioning), use:

```bash
modal run ActorPipeline/modal_two_endpoint_from_keyframes.py --keyframe-run-id <keyframe_run_id> --scenario-path ActorPipeline/squatt_scenario.json
```

This file is a backend skeleton and includes:
- keyframe pair planning
- scenario transition prompt mapping
- segment boundary-safe concatenation
- a backend hook (`_generate_segment_with_backend`) where you integrate a model that supports first+last frame conditioning

Suggested first backend to wire: **Wan 2.1 FLF2V** (first/last frame image-to-video).
