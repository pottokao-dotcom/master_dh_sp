#!/bin/bash
cd ~/master_dh_sp; P=~/ComfyUI_bh/.venv/bin/python; L8=qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928-ck1174-f16.gguf
until grep -q EXIT examples/v5_boogu.log; do sleep 10; done
(for sd in 42 7; do $P tools/make_examples.py --jobs examples/picks_s$sd.json --seeds $sd --tag v5_ideo4_s$sd --dit ideo4 --te ideogram4-te-heretic-dhq-Q4_K_M.gguf --lora $L8 --draft none --comfy 127.0.0.1:8196; done > examples/v5_ideo4.log 2>&1; echo EXIT $? >> examples/v5_ideo4.log)
