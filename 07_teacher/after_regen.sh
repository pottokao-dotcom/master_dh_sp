#!/bin/bash
set -u
T=~/teacher_v4/07_teacher; L=~/teacher_v4/after_regen.log
until grep -q "regen done" ~/teacher_v4/regen_design.log 2>/dev/null; do sleep 60; done
grep -q "regen done: DONE" ~/teacher_v4/regen_design.log || { echo "$(date +%T) REGEN DID NOT FINISH (no DONE), stop" > ~/teacher_v4/after_regen.log; exit 1; }
echo "$(date '+%T') regen done" > $L
cd $T && ~/nunchaku-venv/bin/python audit.py accepted.jsonl 500 > audit_full.txt 2>&1
~/nunchaku-venv/bin/python ~/teacher_v4/gate.py >> $L 2>&1 || { echo "$(date '+%T') AUDIT GATE FAILED" >> $L; exit 1; }
(cd ~/teacher_v4 && sha256sum -c --quiet freeze_v4.sha256) >> $L 2>&1 || { echo "$(date '+%T') FILES != v4-freeze, stop" >> $L; exit 1; }
~/nunchaku-venv/bin/python ~/teacher_v4/fullrebuild.py >> $L 2>&1 || { echo "$(date '+%T') FULL REBUILD MISMATCH, stop" >> $L; exit 1; }
~/nunchaku-venv/bin/python to_dataset.py accepted.jsonl >> $L 2>&1
~/train-venv/bin/python - >> $L 2>&1 <<'PY' || { echo "$(date '+%T') LENGTH CHECK FAILED" >> $L; exit 1; }
import json
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("/home/pottokao/qi21_te_heretic_hf", trust_remote_code=True)
m=0
for l in open("/home/pottokao/teacher_v4/07_teacher/dataset_v4_8b.jsonl"):
    r=json.loads(l); p=tok.apply_chat_template(r["messages"][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    n=len(tok(p,add_special_tokens=False)["input_ids"])+len(tok(r["messages"][2]["content"]+(tok.eos_token or ""),add_special_tokens=False)["input_ids"]); m=max(m,n)
print("8B max tokens",m); import sys; sys.exit(0 if m<=4352 else 1)
PY
# 換上新節點包(新版面卡＋複合招牌句),真 ComfyUI 節點(CPU)逐字比對新訓練料 system/user
mv ~/mdh/ComfyUI-MasterStyle ~/mdh/ComfyUI-MasterStyle.v4old && mv ~/mdh/ComfyUI-MasterStyle.new ~/mdh/ComfyUI-MasterStyle
rm -f ~/teacher_v4/node_io_check.log; bash ~/teacher_v4/node_io_run.sh
cat ~/teacher_v4/node_io_check.log >> $L
[ $(grep -c "一樣 150 / 150" ~/teacher_v4/node_io_check.log) = 2 ] || { echo "$(date '+%T') NODE INPUT MISMATCH, stop" >> $L; exit 1; }
mv ~/teacher_v4/contam_render ~/teacher_v4/contam_render_r1_oldsig
docker stop gemma26m >> $L 2>&1
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}'); [ "$u" -lt 1000 ] && break; sleep 2; done
bash ~/teacher_v4/contam_render.sh
echo "$(date '+%T') 8B smoke (5 steps)" >> $L
cd ~/teacher_v4/08_train && MAX_STEPS=5 LOG_STEPS=1 SAVE_STEPS=100000 DATA_PATH=$T/dataset_v4_8b.jsonl OUT_DIR=$HOME/teacher_v4/adapter_8b_smoke \
  ~/train-venv/bin/accelerate launch --config_file ~/lora_pe_src/ddp2.yaml train_v4_8b_pack.py 2 > ~/teacher_v4/train_8b_smoke.log 2>&1
PEAK=$(grep -o "保留 [0-9.]* GiB" ~/teacher_v4/train_8b_smoke.log | awk '{print $2}' | sort -n | tail -1)
echo "$(date '+%T') smoke exit, peak reserved ${PEAK:-none} GiB" >> $L
[ -n "$PEAK" ] && awk "BEGIN{exit !($PEAK < 15.0)}" || { echo "$(date '+%T') SMOKE FAILED or too close to 16G, stop" >> $L; exit 1; }
rm -rf ~/teacher_v4/adapter_8b_smoke
(cd ~/teacher_v4 && sha256sum -c --quiet freeze_v4.sha256) >> $L 2>&1 || { echo "$(date '+%T') FILES != v4-freeze before training, stop" >> $L; exit 1; }
mkdir -p ~/teacher_v4/adapter_8b && cp ~/teacher_v4/freeze_v4.sha256 ~/teacher_v4/adapter_8b/ && sha256sum $T/dataset_v4_8b.jsonl > ~/teacher_v4/adapter_8b/dataset.sha256
echo "$(date '+%T') start 8B training" >> $L
DATA_PATH=$T/dataset_v4_8b.jsonl OUT_DIR=$HOME/teacher_v4/adapter_8b ~/train-venv/bin/accelerate launch --config_file ~/lora_pe_src/ddp2.yaml train_v4_8b_pack.py 2 > ~/teacher_v4/train_8b.log 2>&1
echo "$(date '+%T') 8B training exit $?" >> $L
