# BloodCodeVideoGen
Generisanje videa za BloodCode nutricionističku aplikaciju.

## Video model comparison (Modal)

Script za poređenje tri text-to-video modela (HunyuanVideo, Wan2.1, LTX) na isti prompt. Inferenca se radi na Modal-u (remote GPU).

### Zahtevi
- [Modal](https://modal.com) nalog i `modal token set` (ili env `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`)
- Python 3.11+

### Instalacija
```bash
pip install modal
modal setup  # ako još niste
```

### Ulaz (JSON)
- **prompt** (obavezno): tekst opisa videa
- **models** (opciono): lista `["hunyuan", "wan", "ltx"]` – podrazumevano sva tri
- **seed** (opciono): za reprodukciju
- **duration_seconds** (opciono): trajanje u sekundama (~10–12 s podrazumevano)

### Pokretanje

**Brzi Hunyuan (1.5, 480p, 12 koraka, 2× upscale)** – za poređenje sa originalnim Hunyuanom, izlaz u posebnom folderu + meta.json:
```bash
cd BloodCodeVideoGen
modal run modal_video_compare.py::app.main_detached --input-path prompt_example.json --model hunyuan_fast
```
Izlaz: `my-volume/hunyuan_fast/<run_id>/hunyuan_fast.mp4` i `meta.json` (model, distilled, rezolucija, koraci, itd.).

**Jedan model, detached** (hunyuan, wan, ltx ili hunyuan_fast):
```bash
modal run modal_video_compare.py::app.main_detached --input-path prompt_example.json --model hunyuan
```
Izlaz: `my-volume/<run_id>/<model>.mp4`. Logovi u Modal dashboardu.

**Sva tri modela** (CLI čeka da sve tri završe):
```bash
modal run modal_video_compare.py --input-path prompt_example.json
```
Izlazni MP4 na volumenu: `my-volume/<run_id>/hunyuan.mp4`, `wan.mp4`, `ltx.mp4`.

### Primer iz stdin (detached, jedan model)
```bash
echo '{"prompt": "A cat walking in the rain"}' | modal run modal_video_compare.py::app.main_detached --input-path - --model hunyuan
```

