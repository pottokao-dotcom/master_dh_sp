#!/bin/bash
# v5b 時尚海報/美妝封面(攝影卡+版面卡複合)。GPU0 等 v5_zbt 完 → zbt → klein;GPU1 等 v5_qi21 完 → qi21
cd ~/master_dh_sp; P=~/ComfyUI_bh/.venv/bin/python
(until grep -q EXIT examples/v5_zbt.log; do sleep 10; done
 $P tools/make_examples.py --jobs examples/v5b.json --tag v5b_zbt --dit zbt --comfy 127.0.0.1:8195 > examples/v5b_zbt.log 2>&1; echo EXIT $? >> examples/v5b_zbt.log
 $P tools/make_examples.py --jobs examples/v5b.json --tag v5b_klein --dit klein --comfy 127.0.0.1:8195 > examples/v5b_klein.log 2>&1; echo EXIT $? >> examples/v5b_klein.log) &
(until grep -q EXIT examples/v5_qi21.log 2>/dev/null; do sleep 10; done
 $P tools/make_examples.py --jobs examples/v5b.json --tag v5b_qi21 --dit qi21 --te qi21-te-heretic-dhq-Q4_K_M.gguf --lora qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928-ck1174-f16.gguf --draft none --comfy 127.0.0.1:8196 > examples/v5b_qi21.log 2>&1; echo EXIT $? >> examples/v5b_qi21.log) &
wait
