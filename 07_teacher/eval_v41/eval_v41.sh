#!/bin/bash
# v4.1 檢查點評估:等鏈跑完 → 6 個檢查點轉 f16 GGUF → 兩台 ComfyUI(GPU0 = 8B、GPU1 = 4B)→ heldout 子集 120 筆真節點擴寫(不出圖)→ 評分。
set -u
T=~/teacher_v41; E=$T/eval; L=$E/eval.log; mkdir -p $E $T/gguf
log(){ echo "$(date '+%F %T') $*" >> $L; }
until grep -qE "CHAIN DONE|4B FAILED|, stop" <(awk '/chain armed/{buf=""} {buf=buf$0"\n"} END{printf buf}' $T/chain.log); do sleep 60; done
grep -q "CHAIN DONE" $T/chain.log || { log "chain not done, no eval"; exit 1; }
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}'); [ "$u" -lt 1000 ] && break; sleep 2; done
log "chain done, start eval"

# 1 轉 GGUF(命名:<底模>_<用途>_lora_<資料版>-r<rank>-ep<epoch>_<日期>-ck<step>-<精度>)
L4=$(ls -d $T/adapter_4b/checkpoint-* | sort -t- -k2 -n | tail -3); L8=$(ls -d $T/adapter_8b/checkpoint-* | sort -t- -k2 -n | tail -3)
log "4B ckpts: $(echo $L4 | xargs -n1 basename | tr '\n' ' ') | 8B ckpts: $(echo $L8 | xargs -n1 basename | tr '\n' ' ')"
conv(){ # $1 adapter dir, $2 base, $3 out
  cd ~/llamacpp_conv && PYTHONPATH=$HOME/llamacpp_conv:$HOME/llamacpp_conv/gguf-py $HOME/train-venv/bin/python convert_lora_to_gguf.py "$1" --base "$2" --outfile "$3" --outtype f16 >> $E/convert.log 2>&1 || { log "CONVERT FAILED $1"; exit 1; }
  ln -sf "$3" ~/ComfyUI_lab/models/loras/mdh/$(basename "$3"); }
for c in $L4; do s=${c##*-}; conv $c $HOME/zeng_v6 $T/gguf/zeng-v6_master-pe_lora_v4.1-r32-ep2_20260928-ck$s-f16.gguf; done
for c in $L8; do s=${c##*-}; conv $c $HOME/qi21_te_heretic_hf $T/gguf/qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928-ck$s-f16.gguf; done
(cd $T/gguf && sha256sum *.gguf > SHA256SUMS); log "converted: $(ls $T/gguf/*.gguf | wc -l) files"

# 2 heldout 子集
~/nunchaku-venv/bin/python $E/make_in.py $T/07_teacher/heldout_v41_4b.jsonl ~/mdh/eval41_4b_in.json >> $L 2>&1
~/nunchaku-venv/bin/python $E/make_in.py $T/07_teacher/heldout_v41_8b.jsonl ~/mdh/eval41_8b_in.json >> $L 2>&1

# 3 兩台 ComfyUI
cd ~/ComfyUI_lab
for g in 0 1; do p=$((8189+g)); CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$HOME/lab_extra_x86 LD_LIBRARY_PATH=$HOME/mdh/cuda133/lib setsid -f ~/nunchaku-venv/bin/python main.py --listen 127.0.0.1 --port $p --disable-auto-launch > ~/mdh/comfy_gpu$g.log 2>&1; done
for p in 8189 8190; do for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:$p/object_info/DualHeadRewrite && break; sleep 3; done; done
log "comfy up"

# 4 擴寫(每台依序跑三個檢查點)
run8(){ for c in $L8; do s=${c##*-}; NOIMG=1 LORA=mdh/qwen3vl-8b-heretic_master-pe_lora_v4.1-r32-ep2_20260928-ck$s-f16.gguf \
  ~/nunchaku-venv/bin/python $E/mdh3_v41.py ~/mdh/eval41_8b_in.json $E/8b_ck$s qwen_image 8189 1.0 > $E/8b_ck$s.log 2>&1; echo "$(date '+%F %T') 8B ck$s done" >> $L; done; }
run4(){ for c in $L4; do s=${c##*-}; NOIMG=1 LORA=mdh/zeng-v6_master-pe_lora_v4.1-r32-ep2_20260928-ck$s-f16.gguf \
  ~/nunchaku-venv/bin/python $E/mdh3_v41.py ~/mdh/eval41_4b_in.json $E/4b_ck$s zimage 8190 1.0 > $E/4b_ck$s.log 2>&1; echo "$(date '+%F %T') 4B ck$s done" >> $L; done; }
run8 & P8=$!; run4 & P4=$!; wait $P8 $P4

# 5 關 ComfyUI、評分
for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null)
  case "$c" in "/home/pottokao/nunchaku-venv/bin/python main.py --listen 127.0.0.1 --port 8189 "*|"/home/pottokao/nunchaku-venv/bin/python main.py --listen 127.0.0.1 --port 8190 "*) kill ${p#/proc/};; esac; done
A4=""; for c in $L4; do s=${c##*-}; A4="$A4 ck$s=$E/4b_ck$s/res.json"; done
A8=""; for c in $L8; do s=${c##*-}; A8="$A8 ck$s=$E/8b_ck$s/res.json"; done
{ echo "== 4B(V6 dhq_2p9 本體)"; ~/nunchaku-venv/bin/python $E/score_v41.py ~/mdh/eval41_4b_in.json $A4
  echo "== 8B(heretic Q4_K_M 本體)"; ~/nunchaku-venv/bin/python $E/score_v41.py ~/mdh/eval41_8b_in.json $A8; } > $E/score.txt 2>&1
cat $E/score.txt >> $L
log "EVAL DONE"
