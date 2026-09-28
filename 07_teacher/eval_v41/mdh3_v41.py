#!/usr/bin/env python
"""(v4.1 評估用副本:8B 的 LORA/GGUF 也可用環境變數指定)moon 三邊校正:雙頭龍(同一顆 Q4_K_M 擴寫+編碼)出圖。
python3 mdh3.py <in.json> <out_dir> <engine zimage|qwen_image> <port> <scale> [<scale> ...]
in.json:[{id, system, user, negative}];擴寫只掛 rewrite LoRA,編碼不掛。"""
import json, math, os, re, shutil, sys, time, urllib.request, urllib.error
from pathlib import Path
INP, OUT, ENGINE, PORT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
SCALES = [float(x) for x in sys.argv[5:]]
C = f"http://127.0.0.1:{PORT}"
OUT.mkdir(parents=True, exist_ok=True)
OUTDIR = Path.home() / "ComfyUI_lab/output"
SEED = 42
if ENGINE == "zimage":
    GGUF, LORA, TYPE = os.environ.get("GGUF", "mdh/zimage_engineer_v6_dhq_2p9.gguf"), os.environ.get("LORA", "mdh/qwen3_4b_bagu_lora_v3-f16.gguf"), "lumina2"
else:
    GGUF, LORA, TYPE = os.environ.get("GGUF", "mdh/qwen3vl_8b_heretic-Q4_K_M.gguf"), os.environ.get("LORA", "mdh/qi21_bagu_lora_v3-f16.gguf"), "qwen_image"

def loader():
    return {"class_type": "DualHeadDragonLoader", "inputs": {"gguf": GGUF, "mmproj": "none", "n_ctx": 32768, "type": TYPE}}

def run(name, g, timeout=1800):
    req = urllib.request.Request(C + "/prompt", json.dumps({"prompt": g}).encode(), {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]
    except urllib.error.HTTPError as e:
        print(name, "SUBMIT_ERR", e.read().decode()[:1500], flush=True); return None
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        try:
            h = json.load(urllib.request.urlopen(C + "/history/" + pid, timeout=10))
        except Exception:
            continue
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        print(name, "ERROR", m[1].get("exception_message", "")[:1500], flush=True)
                return None
            print(f"{name:34} {time.time()-t0:4.0f}s", flush=True)
            return h[pid]["outputs"]
    print(name, "TIMEOUT", flush=True)

def size(ratio, area=1024 * 1024):
    try:
        a, b = (float(x) for x in ratio.split(":")); r = a / b
    except Exception:
        return 1024, 1024
    w = math.sqrt(area * r)
    return int(round(w / 16) * 16), int(round(w / r / 16) * 16)

def rewrite(it, scale):
    return {"CL": loader(),
            "RW": {"class_type": "DualHeadRewrite", "inputs": {
                "clip": ["CL", 0], "prompt": it["user"], "system_prompt": "none", "system_prompt_text": it["system"],
                "rewrite_lora": LORA if scale > 0 else "none", "lora_strength": max(scale, 0.05), "seed": SEED,
                "max_new_tokens": 2048, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "thinking": False}},
            "T": {"class_type": "PreviewAny", "inputs": {"source": ["RW", 0]}},
            "R": {"class_type": "PreviewAny", "inputs": {"source": ["RW", 1]}},
            "W": {"class_type": "PreviewAny", "inputs": {"source": ["RW", 2]}}}

def image_zimage(text, w, h, prefix):
    k = lambda m, lat, a, b, noise: {"class_type": "KSamplerAdvanced", "inputs": {
        "model": m, "positive": ["E", 0], "negative": ["N", 0], "latent_image": lat, "add_noise": noise,
        "noise_seed": SEED, "steps": 12, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple",
        "start_at_step": a, "end_at_step": b, "return_with_leftover_noise": "enable" if b < 12 else "disable"}}
    return {"UB": {"class_type": "UNETLoader", "inputs": {"unet_name": "mdh/z-image-base-nvfp4_ultra.safetensors", "weight_dtype": "default"}},
            "L": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["UB", 0], "lora_name": "mdh/Z-Image-Fun-Lora-Distill-8-Steps-2603-ComfyUI.safetensors", "strength_model": 0.8}},
            "MB": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["L", 0], "shift": 3.0}},
            "UT": {"class_type": "UNETLoader", "inputs": {"unet_name": "mdh/z_image_turbo_nvfp4.safetensors", "weight_dtype": "default"}},
            "MT": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["UT", 0], "shift": 3.0}},
            "V": {"class_type": "VAELoader", "inputs": {"vae_name": "mdh/ae.safetensors"}},
            "CL": loader(),
            "E": {"class_type": "CLIPTextEncodeLumina2", "inputs": {"clip": ["CL", 0], "system_prompt": "superior", "user_prompt": text}},
            "N": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["E", 0]}},
            "EL": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "K1": k(["MB", 0], ["EL", 0], 0, 2, "enable"),
            "K2": k(["MT", 0], ["K1", 0], 2, 12, "disable"),
            "D": {"class_type": "VAEDecode", "inputs": {"samples": ["K2", 0], "vae": ["V", 0]}},
            "S": {"class_type": "SaveImage", "inputs": {"images": ["D", 0], "filename_prefix": prefix}}}

def image_qi21(text, neg, w, h, prefix):
    # 25 步:1–17 步 cfg 1,18–25 步 cfg 3(使用者 2026-09-27 指定)
    k = lambda lat, a, b, cfg, noise: {"class_type": "KSamplerAdvanced", "inputs": {
        "model": ["Ca", 0], "positive": ["E", 0], "negative": ["E", 1], "latent_image": lat, "add_noise": noise,
        "noise_seed": SEED, "steps": 25, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
        "start_at_step": a, "end_at_step": b, "return_with_leftover_noise": "enable" if b < 25 else "disable"}}
    return {"U": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_nvfp4.safetensors", "weight_dtype": "default"}},
            "Ca": {"class_type": "QwenImage21Cache", "inputs": {"model": ["U", 0], "device": "auto", "dtype": "default"}},
            "V": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            "CL": loader(),
            "E": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["CL", 0], "prompt": text, "negative_prompt": neg,
                                                                   "resolution": 1024, "vae": ["V", 0]}},
            "EL": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "K1": k(["EL", 0], 0, 17, 1.0, "enable"),
            "K2": k(["K1", 0], 17, 25, 3.0, "disable"),
            "D": {"class_type": "VAEDecode", "inputs": {"samples": ["K2", 0], "vae": ["V", 0]}},
            "S": {"class_type": "SaveImage", "inputs": {"images": ["D", 0], "filename_prefix": prefix}}}

try:
    from opencc import OpenCC
    _t2s = OpenCC("t2s").convert
except Exception:
    _t2s = None

def guard_quotes(prompt, text):
    for q in re.findall(r"「([^」]+)」|『([^』]+)』|\"([^\"]+)\"|“([^”]+)”", prompt):
        q = next(x for x in q if x)
        if q in text or not _t2s:
            continue
        s = _t2s(q)
        if s != q and s in text:
            text = text.replace(s, q)
    return text

items = json.loads(INP.read_text())
res = {}
for it in items:
    res[it["id"]] = {}
    for sc in SCALES:
        o = run(f"{it['id']}_{sc}_rw", rewrite(it, sc))
        if not o:
            continue
        text, ratio, raw = o["T"]["text"][0], o["R"]["text"][0], o["W"]["text"][0]
        for retry in (1, 2):                            # 保險 2:亂碼就換 seed 重生
            if "\ufffd" not in text:
                break
            g2 = rewrite(it, sc); g2["RW"]["inputs"]["seed"] = SEED + retry
            o = run(f"{it['id']}_{sc}_rw_retry{retry}", g2) or o
            text, ratio, raw = o["T"]["text"][0], o["R"]["text"][0], o["W"]["text"][0]
        text = guard_quotes(it["prompt"], text)       # 保險 1:標題被轉簡體就換回原字
        if os.environ.get("NOIMG"):
            res[it["id"]][str(sc)] = {"text": text, "wh_ratio": ratio, "raw": raw}
            (OUT / "res.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
            continue
        final = it.get("lead", "") + (" " if it.get("lead") else "") + text + (" " + it["close"] if it.get("close") else "")
        w, h = size(ratio)
        pre = f"m3_{ENGINE}_{it['id']}_{sc}"
        g = image_zimage(final, w, h, pre) if ENGINE == "zimage" else image_qi21(final, it.get("negative", ""), w, h, pre)
        o2 = run(f"{it['id']}_{sc}_img {w}x{h}", g)
        f = o2 and [i for n in o2.values() for i in n.get("images", [])]
        dst = None
        if f:
            dst = OUT / f"{it['id']}_{sc}.png"
            shutil.copy(OUTDIR / f[0].get("subfolder", "") / f[0]["filename"], dst)
        res[it["id"]][str(sc)] = {"text": text, "final": final, "wh_ratio": ratio, "file": dst.name if dst else None}
        (OUT / "res.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("DONE", flush=True)
