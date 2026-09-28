#!/bin/bash
# v5 正式一輪:24 題 x seed 42,7。GPU0=base+turbo;GPU1=klein 再接 QI2.1
cd ~/master_dh_sp; P=~/ComfyUI_bh/.venv/bin/python
($P tools/make_examples.py --jobs examples/v5.json --tag v5_zbt --dit zbt --comfy 127.0.0.1:8195 > examples/v5_zbt.log 2>&1; echo EXIT $? >> examples/v5_zbt.log) &
($P tools/make_examples.py --jobs examples/v5.json --tag v5_klein --dit klein --comfy 127.0.0.1:8196 > examples/v5_klein.log 2>&1; echo EXIT $? >> examples/v5_klein.log
 $P tools/make_examples.py --jobs examples/v5.json --tag v5_qi21 --dit qi21 --te qi21-te-heretic-dhq-Q4_K_M.gguf --lora qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928-ck1174-f16.gguf --draft none --comfy 127.0.0.1:8196 > examples/v5_qi21.log 2>&1; echo EXIT $? >> examples/v5_qi21.log) &
wait
