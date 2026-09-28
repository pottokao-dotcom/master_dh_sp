"""klein 9B 範例:直接沿用 klein 4B 那輪的擴寫(同引擎 klein、同 LoRA、同 seed → 擴寫文字相同),
只換 TE(Qwen3-8B fp8mixed,產線同款)和 DiT(klein 9B bf16 → 載入時轉 fp8_e4m3fn;產線是 nvfp4,HF 需授權)。
只跑 v5_picks.json 選的題目與 seed。"""
import json, os, sys, time, urllib.request, uuid
import websocket
E = os.path.expanduser("~/master_dh_sp/examples/")
U = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8196"
TAG = "v5_klein9"
picks = {(p["card"], p["prompt"]): p for p in json.load(open(E + "v5_picks.json"))["picks"]}
src = [r for t in ("v5_klein", "v5b_klein") for r in json.load(open(E + t + "/log.json"))]
jobs = [r for r in src if (r["code"], r["prompt"]) in picks and r["seed"] == picks[(r["code"], r["prompt"])]["seed"]]
cid = str(uuid.uuid4()); ws = websocket.create_connection("ws://%s/ws?clientId=%s" % (U, cid))
def run(g):
    pid = json.load(urllib.request.urlopen(urllib.request.Request("http://%s/prompt" % U,
          json.dumps({"prompt": g, "client_id": cid}).encode(), {"Content-Type": "application/json"})))["prompt_id"]
    while True:
        m = ws.recv()
        if not isinstance(m, str): continue
        m = json.loads(m); d = m.get("data", {})
        if d.get("prompt_id") != pid: continue
        if m["type"] == "execution_error": raise RuntimeError(d.get("exception_message"))
        if m["type"] == "executing" and d["node"] is None: break
    return json.load(urllib.request.urlopen("http://%s/history/%s" % (U, pid)))[pid]["outputs"]
os.makedirs(E + TAG, exist_ok=True); log = []
for i, r in enumerate(jobs):
    t0 = time.time(); w, h = r["size"]; seed = r["seed"]
    g = {"c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}},
         "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": r["final_prompt"], "clip": ["c", 0]}},
         "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
         "u": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-9b.safetensors", "weight_dtype": "fp8_e4m3fn"}},
         "v": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
         "ks": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
         "noi": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
         "lat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
         "sch": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": w, "height": h}},
         "g": {"class_type": "CFGGuider", "inputs": {"model": ["u", 0], "positive": ["pos", 0], "negative": ["neg", 0], "cfg": 1.0}},
         "k": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noi", 0], "guider": ["g", 0], "sampler": ["ks", 0], "sigmas": ["sch", 0], "latent_image": ["lat", 0]}},
         "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
         "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": "examples/%s/%02d_%s_s%d" % (TAG, i, r["code"], seed), "images": ["dec", 0]}}}
    img = run(g)["save"]["images"][0]["filename"]
    log.append({**r, "i": i, "image": img}); json.dump(log, open(E + TAG + "/log.json", "w"), ensure_ascii=False, indent=1)
    print(i, r["code"], "%.1fs" % (time.time() - t0), img, flush=True)
