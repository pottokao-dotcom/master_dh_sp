"""v4.1 上傳(使用者 09-28:不評估,直接轉好丟 HF private)。
4B:最後三個檢查點 → f16 GGUF ＋ adapter → 新 private repo;8B:同三個檢查點補 f16 GGUF 進既有 repo。上傳後比對遠端 sha。"""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from huggingface_hub import HfApi

T = Path.home() / "teacher_v41"; G = T / "gguf"; G.mkdir(exist_ok=True)
REPO4 = "pottokao/zeng-v6_master-pe_lora_v4.1-r32-ep2_20260928"
REPO8 = "pottokao/qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
last3 = lambda d: sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[-3:]
api = HfApi()

def convert(ck, base, out):
    if out.exists(): return
    env = dict(os.environ, PYTHONPATH=f"{Path.home()}/llamacpp_conv:{Path.home()}/llamacpp_conv/gguf-py")
    r = subprocess.run([f"{Path.home()}/train-venv/bin/python", "convert_lora_to_gguf.py", str(ck), "--base", str(base),
                        "--outfile", str(out), "--outtype", "f16"], cwd=Path.home() / "llamacpp_conv", env=env,
                       capture_output=True, text=True)
    open(G / "convert.log", "a").write(r.stdout + r.stderr)
    if r.returncode or not out.exists(): sys.exit(f"CONVERT FAILED {ck}")

def gguf_name(prefix, ck):
    return f"{prefix}_master-pe_lora_v4.1-r32-ep2_20260928-ck{ck.name.split('-')[1]}-f16.gguf"

def verify(repo, files):
    info = {p.path: p for p in api.get_paths_info(repo, list(files))}
    bad = [f for f, h in files.items() if not info.get(f) or not info[f].lfs or info[f].lfs.sha256 != h]
    print(repo, "遠端 sha 一致" if not bad else f"不一致 {bad}", len(files), "檔", flush=True)
    return not bad

# ── 4B ──
A4 = T / "adapter_4b"; C4 = last3(A4)
assert (A4 / "final").exists(), "4B 沒有 final"
S = Path.home() / "hf_stage_4b_v41"; shutil.rmtree(S, ignore_errors=True); S.mkdir()
rows, files4 = [], {}
for ck in C4:
    out = G / gguf_name("zeng-v6", ck); convert(ck, Path.home() / "zeng_v6", out)
    d = S / ck.name; d.mkdir()
    for f in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json"): shutil.copy(ck / f, d / f)
    os.link(out, d / out.name) if out.stat().st_dev == d.stat().st_dev else shutil.copy(out, d / out.name)
    st = json.load(open(ck / "trainer_state.json")); last = [h for h in st["log_history"] if "loss" in h][-1]
    h_ad, h_gg = sha(d / "adapter_model.safetensors"), sha(d / out.name)
    files4[f"{ck.name}/adapter_model.safetensors"] = h_ad; files4[f"{ck.name}/{out.name}"] = h_gg
    rows.append({"name": ck.name, "step": st["global_step"], "epoch": round(st["epoch"], 3), "last_logged_loss": last["loss"],
                 "adapter_sha256": h_ad, "gguf": out.name, "gguf_sha256": h_gg})
same = sha(C4[-1] / "adapter_model.safetensors") == sha(A4 / "final" / "adapter_model.safetensors")
for f in ("freeze_v41.sha256", "dataset_v41.sha256"): shutil.copy(A4 / f, S / f)
man = {"base_model": "Z-Image-Engineer V6(Qwen3-4B 系,moon ~/zeng_v6,Apache-2.0)", "deploy_body": "zimage_engineer_v6_dhq_2p9.gguf",
       "purpose": "master-pe(大師風格擴寫,雙頭龍擴寫頭,Z-Image)", "data": "v4.1c:v4 無版面 14,741 + v4.1 複合 2,131;4B train 8,273 / heldout 294",
       "freeze_tag": "v4.1c-freeze", "hyper": {"method": "QLoRA nf4, DDP x2, shared-prefix pack x4", "r": 32, "alpha": 64, "dropout": 0.05,
       "lr": 1e-4, "sched": "cosine, warmup 3%", "eff_batch": 16, "epochs": 2}, "checkpoints": rows,
       "note": f"最後一個檢查點 {'==' if same else '!='} final。未評估;optimizer 未上傳(留 moon ~/teacher_v41/adapter_4b)。GGUF 為 llama.cpp LoRA(f16),掛在擴寫模式。"}
json.dump(man, open(S / "manifest.json", "w"), ensure_ascii=False, indent=1)
tbl = "\n".join(f"| {r['name']} | {r['step']} | {r['epoch']:.2f} | {r['last_logged_loss']} | `{r['gguf']}` |" for r in rows)
open(S / "README.md", "w").write(f"""# {REPO4.split('/')[1]}

大師風格擴寫 LoRA(4B,Z-Image 雙頭龍擴寫頭),資料 v4.1c。**private · 尚未評估**。

| 檢查點 | step | epoch | 最後紀錄 loss | llama.cpp GGUF |
|---|---|---|---|---|
{tbl}

- 底模:Z-Image-Engineer V6(上線本體 `zimage_engineer_v6_dhq_2p9.gguf`);LoRA 只掛擴寫模式
- QLoRA nf4、r32/alpha64/dropout0.05、lr 1e-4 cosine、有效批次 16、2 epoch、共用前綴打包 ×4
- 資料:v4 無版面 14,741 + v4.1 複合 2,131(4B train 8,273 / heldout 294);凍結 `v4.1c-freeze`
- 各檔 sha256 見 `manifest.json`
""")
api.create_repo(REPO4, private=True, exist_ok=True)
print("4B private:", api.model_info(REPO4).private, flush=True)
api.upload_folder(repo_id=REPO4, folder_path=str(S), commit_message="4B master-pe LoRA v4.1c: last 3 checkpoints (adapter + f16 GGUF)")
ok4 = verify(REPO4, files4)

# ── 8B:補 GGUF ──
A8 = T / "adapter_8b"; C8 = last3(A8); files8 = {}
S8 = Path.home() / "hf_stage_8b_gguf"; shutil.rmtree(S8, ignore_errors=True); S8.mkdir()
for ck in C8:
    out = G / gguf_name("qwen3vl-8b-heretic", ck); convert(ck, Path.home() / "qi21_te_heretic_hf", out)
    (S8 / ck.name).mkdir(); shutil.copy(out, S8 / ck.name / out.name); files8[f"{ck.name}/{out.name}"] = sha(out)
print("8B private:", api.model_info(REPO8).private, flush=True)
api.upload_folder(repo_id=REPO8, folder_path=str(S8), commit_message="add f16 GGUF (llama.cpp LoRA) for checkpoints 1000/1100/1174")
ok8 = verify(REPO8, files8)
(G / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted(G.glob("*.gguf"))))
if ok4 and ok8:
    shutil.rmtree(S); shutil.rmtree(S8); print("staging removed", flush=True)
print("UPLOAD DONE", json.dumps({"4B": files4, "8B": files8}, ensure_ascii=False, indent=1), flush=True)
