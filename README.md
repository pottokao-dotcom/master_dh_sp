# Master Style system (大師風格擴寫系統)

Source of the master-style prompt enhancer: role-play system prompts (`01_system/`), style cards (`02_cards/`, 89
cards), per-DiT engine notes (`03_engines/`), the ComfyUI node and packer (`04_node/`), the teacher pipeline that
wrote the training data (`07_teacher/`), LoRA training (`08_train/`) and the README showcase tools (`tools/`, `examples/`).

- The shipped subset is **Master Style Essentials** (20 cards, `ship_cards.txt`, built by `04_node/pack.py`).
- It runs inside [ComfyUI-DualHeadDragon](https://github.com/pottokao-dotcom/ComfyUI-DualHead-Dragon): one GGUF is both
  the DiT's text encoder and the prompt enhancer, with the master LoRA
  ([4B](https://huggingface.co/pottokao/zeng-v6_master-pe_lora_v4.1-r32-ep2_20260928) /
  [8B](https://huggingface.co/pottokao/qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928)).
- Style cards describe a way of seeing in words only: no photographs were used and no photographer is named. Styles
  after long-deceased painters or schools use that name (`van_gogh_style`, `swiss-international-style`); every other
  style has a descriptive name.
- Design notes: `DESIGN.md`, card rules: `CARD_RULES.md` (Traditional Chinese).

License: not decided yet.
