#!/usr/bin/env python
"""組出可安裝的節點包 dist/ComfyUI-MasterStyle-Essentials(Master Style Essentials / 大師精選):
只含 ship_cards.txt 列的卡;不含 PRIVATE/、00_originals/、測試與備份。node ID 不變,workflow 與完整版通用。"""
import shutil
from pathlib import Path
R = Path(__file__).resolve().parent.parent
EDITION, EDITION_ZH = "Master Style Essentials", "大師精選"
D = R / "dist" / "ComfyUI-MasterStyle-Essentials"
if D.exists():
    shutil.rmtree(D)
(D / "core").mkdir(parents=True)
shutil.copy(R / "04_node/comfy/__init__.py", D / "__init__.py")
shutil.copy(R / "04_node/comfy/t2s_chars.json", D / "t2s_chars.json")
shutil.copy(R / "04_node/master_style.py", D / "core/master_style.py")
(D / "01_system").mkdir(); (D / "02_cards").mkdir(); (D / "03_engines").mkdir()
for f in ("pe_master_roleplay_short.txt", "pe_master_roleplay_long.txt", "pe_master_roleplay_long.multi.txt", "pe_master_roleplay_short.multi.txt", "pe_master_roleplay_short.design.txt", "pe_master_roleplay_long.design.txt", "pe_master_roleplay_long.design_nar.txt", "pe_master_roleplay_short.design_nar.txt", "pe_master_long.txt", "pe_master_short.txt",
          "style_tags.json"):
    shutil.copy(R / "01_system" / f, D / "01_system" / f)
SHIP = [l.split("#")[0].strip() for l in (R / "ship_cards.txt").read_text(encoding="utf-8").splitlines()]
SHIP = [c for c in SHIP if c]                          # 出貨範圍 = ship_cards.txt;不在清單的卡不打包
missing = [c for c in SHIP if not (R / "02_cards" / (c + ".yaml")).exists()]
if missing:
    raise SystemExit("ship_cards.txt 裡有不存在的卡: %s" % missing)
for f in (R / "02_cards" / (c + ".yaml") for c in SHIP):  # 出貨版:拿掉檔頭註解與 source(內含原卡檔名＝人名)
    lines = [l for l in f.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#") and not l.startswith("source:")]
    (D / "02_cards" / f.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
shutil.copy(R / "03_engines/engines.yaml", D / "03_engines/engines.yaml")
(D / "NOTICE").write_text("t2s_chars.json is derived from OpenCC (TSCharacters.txt), Apache License 2.0, "
                          "https://github.com/BYVoid/OpenCC\n")
import sys
sys.path.insert(0, str(D / "core"))
import master_style as MS                               # 用打包後的那份讀卡,README 列的就是實際出貨的
groups = {"default": [], "photo": [], "painting": [], "design": []}
for c in SHIP:
    g = "default" if c == "pe-default" else MS.medium_of(c)
    groups.setdefault(g, []).append("- `%s` %s" % (c, MS.load_card(c).get("label_zh", "")))
readme = ["# %s(%s)" % (EDITION, EDITION_ZH), "",
          "Master-style prompt enhancer for ComfyUI-DualHeadDragon: %d curated styles (%d photography, %d painting, "
          "%d layout) plus the plain default. Each style is a role-play card; the enhancer writes the prompt the way "
          "that style would make the picture." % (len(groups["photo"]) + len(groups["painting"]) + len(groups["design"]),
          len(groups["photo"]), len(groups["painting"]), len(groups["design"])), "",
          "## Wiring", "",
          "Master Style Prompt (system / user) -> Dual-Head Rewrite (system_prompt_text / prompt, master LoRA, optional "
          "DFlash draft) -> Master Style Finalize -> CLIPTextEncode -> DiT. Finalize also returns width / height from "
          "the style's aspect.", "",
          "## Styles", "", "### Default"] + groups["default"] + ["", "### Photography"] + groups["photo"] + ["", "### Painting"] + groups["painting"] + \
         ["", "### Layout"] + groups["design"] + [""]
# 支援的 DiT(擴寫 = 雙頭龍 Dual-Head Rewrite + 大師 LoRA;引擎段依 Loader type 自動接)
readme += ["## Supported DiTs", "",
           "The enhancer runs inside [ComfyUI-DualHeadDragon](https://github.com/pottokao-dotcom/ComfyUI-DualHead-Dragon): one GGUF is both the DiT's text encoder (TE) and the prompt enhancer (PE) with the master LoRA. Every DiT below was run on the showcase prompts.", "",
           "### Support", ""] + ['| DiT | Loader `type` | TE | Rewrite (PE) | Reference images | DFlash | How it was checked |', '|---|---|---|---|---|---|---|', "| Z-Image (turbo, base+turbo) | `lumina2` | yes | yes, 4B master LoRA | n/a | yes (b16) | TE against ComfyUI's own encoder + images |", '| FLUX.2 klein 4B | `flux2` | yes | yes, 4B master LoRA | not done | yes (b16) | same; layout lettering garbled (the DiT) |', "| FLUX.2 klein 9B | `flux2` | yes (needs a Qwen3-8B GGUF; the showcase uses ComfyUI's own TE) | yes, via the 4B rewrite | not done | yes (4B rewrite) | images; no master LoRA for plain Qwen3-8B |", '| Qwen-Image 2.1 | `qwen_image` | yes | yes, 8B master LoRA | yes | no | official bf16 TE comparison + t2i / edit images |', '| Ideogram 4 | `ideogram4` | yes (13 layers) | yes, 8B master LoRA | not checked | no | self-check (read modes agree to 1e-7) + images; official prompts are JSON, plain text here |', '| Boogu-Image Turbo | `boogu` | yes (last layer + norm) | yes, 8B master LoRA | not checked | no | same; most reliable lettering |', '| JoyAI-Image-Edit Plus | `joyimage` | yes (last layer) | yes, 8B master LoRA | no (reference preprocessing differs) | no | images (text to image) |', '', '### Downloads (the files the showcase used)', '', '| DiT | TE / PE GGUF (`models/text_encoders`) | Master LoRA (`models/loras`) | Draft (`models/dualhead_drafts`) | DiT (`models/diffusion_models`) |', '|---|---|---|---|---|', '| Z-Image | [Qwen3-4B-Text-Encoder-DHQ-GGUF](https://huggingface.co/pottokao/Qwen3-4B-Text-Encoder-DHQ-GGUF) `zimage_engineer_v6_dhq_2p9.gguf` (more sizes: [Z-Image-Engineer-V6-DHQ-GGUF](https://huggingface.co/pottokao/Z-Image-Engineer-V6-DHQ-GGUF)) | [4B v4.1](https://huggingface.co/pottokao/zeng-v6_master-pe_lora_v4.1-r32-ep2_20260928) `ck1212` | [V6-DFlash-b16](https://huggingface.co/pottokao/Z-Image-Engineer-V6-DFlash-b16) `Q6_K` | turbo: [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) `z_image_turbo_nvfp4`; base: own `z-image-base-nvfp4_ultra` (not published) + [Distill-8-Steps LoRA](https://huggingface.co/alibaba-pai/Z-Image-Fun-Lora-Distill) |', '| FLUX.2 klein 4B | as Z-Image | as Z-Image | as Z-Image | [black-forest-labs/FLUX.2-klein-4b-nvfp4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4) |', '| FLUX.2 klein 9B | rewrite as Z-Image; TE [Comfy-Org/flux2-klein-9B](https://huggingface.co/Comfy-Org/flux2-klein-9B) `qwen_3_8b_fp8mixed` | as Z-Image | as Z-Image | [black-forest-labs/FLUX.2-klein-9b-nvfp4](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4) (production; gated, accept the FLUX Non-Commercial License on HF). The showcase used the official bf16 [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) loaded as fp8 (same license) |', '| Qwen-Image 2.1 | [Qwen-Image-2.1-Text-Encoder-Heretic-DHQ-GGUF](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-DHQ-GGUF) `Q4_K_M` + `mmproj` | [8B v4.1](https://huggingface.co/pottokao/qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928) `ck1174` | none | own `qwen_image_2.1_nvfp4` (not published) |', '| Ideogram 4 | [Ideogram-4-Text-Encoder-Heretic-DHQ-GGUF](https://huggingface.co/pottokao/Ideogram-4-Text-Encoder-Heretic-DHQ-GGUF) `Q4_K_M` | as QI2.1 | none | [Comfy-Org/Ideogram-4](https://huggingface.co/Comfy-Org/Ideogram-4) `ideogram4_nvfp4_mixed` + `ideogram4_unconditional_nvfp4_mixed` (non-commercial) |', '| Boogu-Image Turbo | [Boogu-Text-Encoder-Heretic-DHQ-GGUF](https://huggingface.co/pottokao/Boogu-Text-Encoder-Heretic-DHQ-GGUF) `Q4_K_M` | as QI2.1 | none | [Comfy-Org/Boogu-Image](https://huggingface.co/Comfy-Org/Boogu-Image) `boogu_image_turbo_hotfix_nvfp4` |', '| JoyAI-Image-Edit Plus | own standard `Q4_K_M` (not uploaded; source [jdopensource/JoyAI-Image-Edit](https://huggingface.co/jdopensource/JoyAI-Image-Edit) `JoyAI-Image-Und/`) | as QI2.1 | none | community [aha2023/JoyAI-Image-Edit-ComfyUI-NVfp4](https://huggingface.co/aha2023/JoyAI-Image-Edit-ComfyUI-NVfp4) `joyai_image_edit_plus_nvfp4` |', '', 'Layout lettering: Boogu is the most reliable, Ideogram 4 and klein 9B are mostly right, JoyAI slips now and then; FLUX.2 klein 4B garbles it (use it for the photography and painting styles).', '']
sys.path.insert(0, str(R / "tools"))
import make_showcase                                     # 範例照:examples/v5_picks.json 選的題目 × 各 DiT
readme += ["## Showcase", "",
           "One prompt per row (left: style and the prompt as typed); every column is the same prompt and seed on another "
           "DiT. Rewrite: temperature 0.7 / top_p 0.8 / top_k 20; each DiT at its official settings (Z-Image base 2 + "
           "turbo 10 steps, klein 4 steps, Qwen-Image 2.1 25 steps, Ideogram 4 Default 20 steps, Boogu 4 steps, JoyAI "
           "40 steps cfg 4). klein 9B DiT loaded as fp8 from bf16; JoyAI DiT is a community NVFP4 "
           "(aha2023/JoyAI-Image-Edit-ComfyUI-NVfp4). Rows with a `+` pair a photography style with a layout style. All showcase images are AI-generated; each DiT has its own model license (Ideogram 4 and FLUX.2 klein 9B are non-commercial) and the images are used under their Outputs terms.", "",
           make_showcase.build(str(D / "showcase"), "showcase/"), ""]
(D / "README.md").write_text("\n".join(readme), encoding="utf-8")
if (R / "release" / "essentials-v1.1.json").exists():  # 版本對照(卡/SP sha、配對的 LoRA 與草稿頭、節點版本)跟著出貨
    shutil.copy(R / "release" / "essentials-v1.1.json", D / "MANIFEST.json")
print(D, sum(1 for _ in D.rglob("*") if _.is_file()), "files,", len(SHIP), "cards shipped")
