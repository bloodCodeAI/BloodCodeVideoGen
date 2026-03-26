"""
Generate 500 confessions per model (base, 1epoch, 2epoch) for engagement alignment evaluation.
Runs on Modal, writes benchmark_generations-style JSON to the volume.
Same prompts and decoding as benchmark_modal.py; no perplexity or test set.

Run:   modal run Test/generate_500_for_alignment_modal.py
Pull:  modal volume get YUGOGPT_training_ispovesti_volume benchmark/
Then:  python Test/eval_alignment.py --generated benchmark/benchmark_generations_500.json --output benchmark/engagement_alignment_report_500.json
"""
import modal
import json
import os

# -----------------------
# SETTINGS (match benchmark_modal.py)
# -----------------------
VOLUME_NAME = "YUGOGPT_training_ispovesti_volume"
BASE_MODEL_HF = "gordicaleksa/YugoGPT"
MODEL_DIR_1EPOCH = "/mnt/models/engagement_models/YugoGPT-engagement-20260128-122111"
MODEL_DIR_2EPOCH = "/mnt/models/engagement_models/YugoGPT-engagement-2EPOCH-ONLY-20260202-193511"
BENCHMARK_DIR = "/mnt/models/benchmark"

NUM_PROMPTS = 500
MAX_NEW_TOKENS = 180
TEMPERATURE = 0.9
TOP_P = 0.95
REPETITION_PENALTY = 1.2
OUTPUT_FILENAME = "benchmark_generations_500.json"

# -----------------------
# MODAL APP
# -----------------------
app = modal.App("YugoGPT-Generate-500-For-Alignment")
volume = modal.Volume.from_name(VOLUME_NAME)


def get_engagement_label(score: float) -> str:
    if score < -0.5:
        return "hated"
    elif score < 0.7:
        return "liked"
    return "loved"


def build_prompts(n: int):
    """Fixed prompts: engagement scores spread from -1 to 1."""
    prompts = []
    for i in range(n):
        score = round(-1.0 + (i / max(1, n - 1)) * 2.0, 4)
        label = get_engagement_label(score)
        prompt = f"[engagement_score:{score}][engagement:{label}] Ispovest:\n"
        prompts.append({"prompt_id": i, "engagement_score": score, "engagement_label": label, "prompt": prompt})
    return prompts


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 3,
    memory=32000,
    volumes={"/mnt/models": volume},
    image=modal.Image.debian_slim().pip_install([
        "torch>=2.1.0",
        "transformers>=4.34.0",
        "peft>=0.5.0",
        "accelerate",
        "sentencepiece",
    ]),
)
def run_generate_500():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    assert MODEL_DIR_1EPOCH != MODEL_DIR_2EPOCH
    assert os.path.exists(MODEL_DIR_1EPOCH), f"1-epoch path not found: {MODEL_DIR_1EPOCH}"
    assert os.path.exists(MODEL_DIR_2EPOCH), f"2-epoch path not found: {MODEL_DIR_2EPOCH}"

    # Tokenizer (from 1-epoch so it has </stop>)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_1EPOCH, use_fast=False)
    stop_token_id = tokenizer.convert_tokens_to_ids("</stop>")
    if stop_token_id == tokenizer.unk_token_id:
        stop_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompts_list = build_prompts(NUM_PROMPTS)
    generations = []

    def generate_with_model(model, model_name, eos_id):
        model.eval()
        for p in prompts_list:
            prompt = p["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            full = tokenizer.decode(out[0], skip_special_tokens=False)
            has_stop = "</stop>" in full
            if "</stop>" in full:
                full = full.split("</stop>")[0]
            full = full.replace("</s>", "").strip()
            if "Ispovest:\n" in full:
                confession = full.split("Ispovest:\n", 1)[1]
            else:
                confession = full[len(prompt):] if full.startswith(prompt) else full
            confession = confession.replace("<s>", "").strip()
            n_tok = out.shape[1] - inputs.input_ids.shape[1]
            generations.append({
                "model": model_name,
                "prompt_id": p["prompt_id"],
                "engagement_score": p["engagement_score"],
                "prompt": prompt,
                "output_text": confession,
                "num_tokens": n_tok,
                "num_chars": len(confession),
                "has_stop": has_stop,
            })
        print(f"  Generated {NUM_PROMPTS} samples for {model_name}")

    # Base
    print("Generating: base...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_HF, torch_dtype=torch.float16, device_map="auto")
    base_model.resize_token_embeddings(len(tokenizer))
    generate_with_model(base_model, "base", tokenizer.eos_token_id)
    del base_model
    torch.cuda.empty_cache()

    # 1-epoch
    print("Generating: 1-epoch...")
    m1 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_HF, torch_dtype=torch.float16, device_map="auto")
    m1.resize_token_embeddings(len(tokenizer))
    m1 = PeftModel.from_pretrained(m1, MODEL_DIR_1EPOCH)
    generate_with_model(m1, "1epoch", stop_token_id)
    del m1
    torch.cuda.empty_cache()

    # 2-epoch
    print("Generating: 2-epoch...")
    m2 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_HF, torch_dtype=torch.float16, device_map="auto")
    m2.resize_token_embeddings(len(tokenizer))
    m2 = PeftModel.from_pretrained(m2, MODEL_DIR_2EPOCH)
    generate_with_model(m2, "2epoch", stop_token_id)
    del m2
    torch.cuda.empty_cache()

    out_path = f"{BENCHMARK_DIR}/{OUTPUT_FILENAME}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(generations, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(generations)} generations to {out_path}")
    volume.commit()
    print("Done. Pull with: modal volume get " + VOLUME_NAME + " benchmark/")
    return {"num_generations": len(generations), "path": out_path}


@app.local_entrypoint()
def main():
    print(f"Generating {NUM_PROMPTS} confessions per model (base, 1epoch, 2epoch) on Modal...")
    app.deploy()
    run_generate_500.remote()
