"""README 範例照:用出貨版(dist/ComfyUI-MasterStyle-Essentials)的 master_style 組 prompt,
走 Dual-Head Rewrite -> finalize -> Z-Image,每張存圖＋擴寫文字。只做 list 裡指定的 (卡, 題目)。"""
import argparse, json, os, sys, time, urllib.request, uuid
sys.path.insert(0, os.path.expanduser("~/repos/ComfyUI-DualHeadDragon/tools"))
sys.path.insert(0, os.path.expanduser("~/master_dh_sp/dist/ComfyUI-MasterStyle-Essentials/core"))
import master_style as MS
from bench_master import dit_graph
# 約 1 MP、邊長 16 的倍數(bench_master 的 DIM 只有 1:1/2:3/3:2,其他比例會掉回正方形)
DIM = {"1:1": (1024, 1024), "2:3": (832, 1248), "3:2": (1248, 832), "3:4": (880, 1184), "4:3": (1184, 880),
       "4:5": (912, 1136), "5:4": (1136, 912), "9:16": (768, 1360), "16:9": (1360, 768)}
BOOGU_ASPECT = {"4:5": "3:4", "5:4": "4:3"}  # Boogu 只收 1:1 2:3 3:2 3:4 4:3 1:2 2:1 9:16 16:9
import websocket

JOY_BUCKETS = [(512, 1792), (512, 1856), (512, 1920), (512, 1984), (512, 2048), (576, 1600), (576, 1664), (576, 1728), (576, 1792),
               (640, 1472), (640, 1536), (640, 1600), (704, 1344), (704, 1408), (704, 1472), (768, 1216), (768, 1280), (768, 1344),
               (832, 1152), (832, 1216), (896, 1088), (896, 1152), (960, 1024), (960, 1088), (1024, 960), (1024, 1024),
               (1088, 896), (1088, 960), (1152, 832), (1152, 896), (1216, 768), (1216, 832), (1280, 768), (1344, 704),
               (1344, 768), (1408, 704), (1472, 640), (1472, 704), (1536, 640), (1600, 576), (1600, 640), (1664, 576),
               (1728, 576), (1792, 512), (1792, 576), (1856, 512), (1920, 512), (1984, 512), (2048, 512)]  # (h, w),comfy nodes_joyimage


def joy_graph(w, h, text, seed, prefix):
    """JoyAI-Image-Edit(Plus)純文字出圖:Comfy 官方藍圖「Image Edit (Joy Image Edit)」設定 40 步 cfg 4 euler/normal + CFGNorm(pre_cfg),
    負向 = 空白 prompt,Wan2.1 VAE;尺寸取官方 1K 分桶中比例最近的一格。"""
    bh, bw = min(JOY_BUCKETS, key=lambda hw: abs(hw[0] / hw[1] - h / w))
    return {"pos": {"class_type": "TextEncodeJoyImageEdit", "inputs": {"clip": ["c", 0], "prompt": text}},
            "neg": {"class_type": "TextEncodeJoyImageEdit", "inputs": {"clip": ["c", 0], "prompt": ""}},
            "u": {"class_type": "UNETLoader", "inputs": {"unet_name": "joyai_image_edit_plus_nvfp4.safetensors", "weight_dtype": "default"}},
            "cn": {"class_type": "CFGNorm", "inputs": {"model": ["u", 0], "strength": 1.0, "pre_cfg": True}},
            "lat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": bw, "height": bh, "batch_size": 1}},
            "k": {"class_type": "KSampler", "inputs": {"model": ["cn", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                  "latent_image": ["lat", 0], "seed": seed, "steps": 40, "cfg": 4.0, "sampler_name": "euler",
                  "scheduler": "normal", "denoise": 1.0}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["dec", 0]}}}


def ideo4_graph(w, h, text, seed, prefix):
    """Ideogram 4 官方 t2i「Default」:20 步 euler、Ideogram4Scheduler mu 0 std 1.75、DualModelGuider cfg 7
    (正向=本體 + CFGOverride 最後 30% 改 cfg 3,負向=unconditional 模型 + 歸零 cond)。Comfy-Org workflow image_ideogram4_t2i.json。"""
    return {"pos": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["c", 0]}},
            "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
            "u": {"class_type": "UNETLoader", "inputs": {"unet_name": "ideogram4_nvfp4_mixed.safetensors", "weight_dtype": "default"}},
            "un": {"class_type": "UNETLoader", "inputs": {"unet_name": "ideogram4_unconditional_nvfp4_mixed.safetensors", "weight_dtype": "default"}},
            "co": {"class_type": "CFGOverride", "inputs": {"model": ["u", 0], "cfg": 3.0, "start_percent": 0.7, "end_percent": 1.0}},
            "g": {"class_type": "DualModelGuider", "inputs": {"model": ["co", 0], "positive": ["pos", 0], "cfg": 7.0,
                  "model_negative": ["un", 0], "negative": ["neg", 0]}},
            "sch": {"class_type": "Ideogram4Scheduler", "inputs": {"steps": 20, "width": w, "height": h, "mu": 0.0, "std": 1.75}},
            "ks": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "noi": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "lat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "k": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noi", 0], "guider": ["g", 0], "sampler": ["ks", 0],
                  "sigmas": ["sch", 0], "latent_image": ["lat", 0]}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
            "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["dec", 0]}}}


def boogu_graph(w, h, text, seed, prefix):
    """Boogu-Image Turbo(hotfix)官方:4 步 lcm / sgm_uniform cfg 1、負向歸零、Flux1 VAE、只支援 1K。"""
    return {"pos": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["c", 0]}},
            "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
            "u": {"class_type": "UNETLoader", "inputs": {"unet_name": "boogu_image_turbo_hotfix_nvfp4.safetensors", "weight_dtype": "default"}},
            "lat": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "k": {"class_type": "KSampler", "inputs": {"model": ["u", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                  "latent_image": ["lat", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "lcm",
                  "scheduler": "sgm_uniform", "denoise": 1.0}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
            "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["dec", 0]}}}


def qi21_graph(w, h, text, neg, seed, prefix):
    """QI2.1 官方 t2i:NVFP4 DiT 25 步 euler/simple cfg 1(同 ~/qi21_t2i_test.py)。"""
    return {"enc": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["c", 0], "prompt": text,
                    "negative_prompt": neg, "resolution": 1024}},
            "u": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_nvfp4.safetensors", "weight_dtype": "default"}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            "lat": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "k": {"class_type": "KSampler", "inputs": {"model": ["u", 0], "positive": ["enc", 0], "negative": ["enc", 1],
                  "latent_image": ["lat", 0], "seed": seed, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                  "scheduler": "simple", "denoise": 1.0}},
            "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["dec", 0]}}}


def zbt_graph(w, h, text, seed, prefix):
    """Z-Image base 2 步(+Distill-8 LoRA 0.8)交棒 turbo 10 步,res_multistep/simple shift 3 cfg 1,共 12 步
    (= 攝影大師 gx10 workflow zimage_base2_turbo8_gx10_nvfp4all.json)。"""
    ks = lambda m, add, a, b, lat, left: {"class_type": "KSamplerAdvanced", "inputs": {"model": m, "add_noise": add,
        "noise_seed": seed, "steps": 12, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple",
        "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": lat, "start_at_step": a, "end_at_step": b,
        "return_with_leftover_noise": left}}
    return {"pos": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["c", 0]}},
            "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
            "ub": {"class_type": "UNETLoader", "inputs": {"unet_name": "z-image-base-nvfp4_ultra.safetensors", "weight_dtype": "default"}},
            "lo": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["ub", 0], "strength_model": 0.8,
                   "lora_name": "Z-Image-Fun-Lora-Distill-8-Steps-2603-ComfyUI.safetensors"}},
            "msb": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lo", 0], "shift": 3.0}},
            "ut": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_nvfp4.safetensors", "weight_dtype": "default"}},
            "mst": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["ut", 0], "shift": 3.0}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
            "lat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "k1": ks(["msb", 0], "enable", 0, 2, ["lat", 0], "enable"),
            "k": ks(["mst", 0], "disable", 2, 12, ["k1", 0], "disable"),
            "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["dec", 0]}}}

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True, help="json: [[code, prompt], ...]")
ap.add_argument("--tag", required=True)
ap.add_argument("--dit", default="zimage", choices=["zimage", "zbt", "klein", "qi21", "ideo4", "boogu", "joy"])
ap.add_argument("--te", default="zimage_engineer_v6_dhq_2p9.gguf")
ap.add_argument("--lora", default="zeng-v6_master-pe_lora_v4.1-r32-ep2_20260928-ck1212-f16.gguf")
ap.add_argument("--draft", default="zeng-v6_master-pe_dflash-b16_run1-Q6_K.gguf")
ap.add_argument("--seeds", default="42,7")
ap.add_argument("--only", default="", help="comma-separated card codes to run (default all)")
ap.add_argument("--comfy", default="127.0.0.1:8195")
a = ap.parse_args()
out = os.path.expanduser("~/master_dh_sp/examples/" + a.tag); os.makedirs(out, exist_ok=True)
U, cid = a.comfy, str(uuid.uuid4())
ws = websocket.create_connection("ws://%s/ws?clientId=%s" % (U, cid))
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
loader = {"class_type": "DualHeadDragonLoader", "inputs": {"gguf": a.te, "mmproj": "none", "n_ctx": 8192,
          "type": {"klein": "flux2", "qi21": "qwen_image", "ideo4": "ideogram4", "boogu": "boogu", "joy": "joyimage"}.get(a.dit, "lumina2")}}
log = []
JOBS = [(j[0], j[1], j[2] if len(j) > 2 else "none", int(sd)) for j in json.load(open(a.jobs)) for sd in a.seeds.split(",")
        if not a.only or j[0] in a.only.split(",")]   # [卡, 題目, 版面卡(選填,複合)]
for i, (code, task, design, seed) in enumerate(JOBS):
    t0 = time.time()
    system, user, neg, aspect = MS.build(task, code=code, engine={"klein": "klein", "qi21": "qwen_image", "ideo4": "qwen_image", "boogu": "qwen_image", "joy": "qwen_image"}.get(a.dit, "zimage"), fmt="roleplay2", seed=seed, design=design)
    w, h = DIM.get(BOOGU_ASPECT.get(aspect, aspect) if a.dit == "boogu" else aspect, (1024, 1024))
    o = run({"c": loader, "r": {"class_type": "DualHeadRewrite", "inputs": {"clip": ["c", 0], "prompt": user,
            "system_prompt": "none", "system_prompt_text": system, "rewrite_lora": a.lora, "lora_strength": 1.0,
            "seed": seed, "max_new_tokens": 2048, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "thinking": False,
            "draft": a.draft, "draft_n_max": 15}}, "show": {"class_type": "PreviewAny", "inputs": {"source": ["r", 0]}}})["show"]
    rewrite = (o.get("text") or o.get("string") or [""])[0]
    final = MS.finalize(rewrite, code=code, prompt=task, design=design)
    pre = "examples/%s/%02d_%s_s%d" % (a.tag, i, code if design == "none" else code + "+" + design, seed)
    if a.dit in ("ideo4", "boogu", "joy"):
        g = {"ideo4": ideo4_graph, "boogu": boogu_graph, "joy": joy_graph}[a.dit](w, h, final, seed, pre); g["c"] = loader
    elif a.dit == "qi21":
        g = qi21_graph(w, h, final, neg or "", seed, "examples/%s/%02d_%s_s%d" % (a.tag, i, code, seed)); g["c"] = loader
    else:
      g = (zbt_graph if a.dit == "zbt" else lambda *x: dit_graph(a.dit, *x))(w, h, final, seed, "examples/%s/%02d_%s_s%d" % (a.tag, i, code, seed)); g["c"] = loader
    img = run(g)["save"]["images"][0]["filename"]
    log.append({"i": i, "code": code if design == "none" else code + "+" + design, "seed": seed, "prompt": task, "aspect": aspect, "size": [w, h], "image": img, "final_prompt": final})
    print(i, code, aspect, "%.1fs" % (time.time() - t0), img, flush=True)
    json.dump(log, open(out + "/log.json", "w"), ensure_ascii=False, indent=1)
