#!/bin/bash
# v4.1 過夜鏈(使用者 09-27 定:資料備好 → 先訓 8B → 再訓 V6 4B,打包 ×4)。任何檢查不符就停並寫 log。
set -u
T=~/teacher_v41; D=$T/07_teacher; L=$T/chain.log
log(){ echo "$(date '+%F %T') $*" >> $L; }
gpu_free(){ for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}'); [ "$u" -lt 1000 ] && return 0; sleep 2; done; return 1; }
log "chain armed, waiting for generation"
until grep -qE "gemma stopped|FAILED|MISMATCH" $T/run.log 2>/dev/null; do sleep 60; done
grep -q "driver: DONE" $T/run.log && grep -q "freeze still ok" $T/run.log || { log "GEN NOT CLEAN, stop"; exit 1; }
log "gen done: $(grep 'driver:' $T/run.log)"
gpu_free || { log "GPU NOT FREE after gen, stop"; exit 1; }

# 1 組合 v4 無版面 + v4.1 複合,全量重組必須 100% 一樣
~/nunchaku-venv/bin/python $T/merge_check.py >> $L 2>&1 || { log "MERGE/REBUILD MISMATCH, stop"; exit 1; }

# 2 轉訓練集(每卡約 3% heldout)
cd $D && ~/nunchaku-venv/bin/python $D/to_dataset_v41.py accepted_all.jsonl >> $L 2>&1 || { log "TO_DATASET FAILED"; exit 1; }
sha256sum dataset_v41_4b.jsonl dataset_v41_8b.jsonl heldout_v41_4b.jsonl heldout_v41_8b.jsonl > $T/dataset_v41.sha256

# 3 長度檢查(8B ≤4352、4B ≤4400,同訓練腳本 MAXLEN)
~/train-venv/bin/python - >> $L 2>&1 <<'PY' || { log "LENGTH CHECK FAILED"; exit 1; }
import json, sys
from transformers import AutoTokenizer
bad = 0
for base, fn, lim in (("/home/pottokao/qi21_te_heretic_hf", "dataset_v41_8b.jsonl", 4352), ("/home/pottokao/zeng_v6", "dataset_v41_4b.jsonl", 4400)):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True); m = 0
    for l in open("/home/pottokao/teacher_v41/07_teacher/" + fn):
        r = json.loads(l); p = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        n = len(tok(p, add_special_tokens=False)["input_ids"]) + len(tok(r["messages"][2]["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]); m = max(m, n)
    print(fn, "max tokens", m, "limit", lim); bad += m > lim
sys.exit(bad)
PY

# 4 換 v4.1 節點包 → 真 ComfyUI 節點(CPU)逐字比對訓練料 system/user
[ -d ~/mdh/ComfyUI-MasterStyle.v41 ] || { log "NO v41 NODE PACK, stop"; exit 1; }
mv ~/mdh/ComfyUI-MasterStyle ~/mdh/ComfyUI-MasterStyle.v4_bak && mv ~/mdh/ComfyUI-MasterStyle.v41 ~/mdh/ComfyUI-MasterStyle
cd ~/ComfyUI_lab && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$HOME/lab_extra_x86 setsid -f ~/nunchaku-venv/bin/python main.py --cpu --listen 127.0.0.1 --port 8190 --disable-auto-launch > ~/mdh/comfy_cpu.log 2>&1
for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:8190/object_info/MasterStylePrompt && break; sleep 3; done
~/nunchaku-venv/bin/python $T/node_io_check_v41.py 8190 150 $D > $T/node_io_check.log 2>&1
for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null)
  case "$c" in "/home/pottokao/nunchaku-venv/bin/python main.py --cpu --listen 127.0.0.1 --port 8190 "*) kill ${p#/proc/};; esac; done
cat $T/node_io_check.log >> $L
[ "$(grep -c '一樣 150 / 150' $T/node_io_check.log)" = 2 ] || { log "NODE INPUT MISMATCH, stop"; exit 1; }
(cd $T && sha256sum -c --quiet freeze_v41.sha256) || { log "FILES != v4.1-freeze, stop"; exit 1; }

# 5 8B(heretic):冒煙 5 步 → 全量
cd ~/teacher_v4/08_train
log "8B smoke"
MAX_STEPS=5 LOG_STEPS=1 SAVE_STEPS=100000 DATA_PATH=$D/dataset_v41_8b.jsonl OUT_DIR=$T/adapter_8b_smoke \
  ~/train-venv/bin/accelerate launch --config_file ~/lora_pe_src/ddp2.yaml train_v4_8b_pack.py 2 > $T/train_8b_smoke.log 2>&1
PEAK=$(grep -o "保留 [0-9.]* GiB" $T/train_8b_smoke.log | awk '{print $2}' | sort -n | tail -1)
log "8B smoke peak reserved ${PEAK:-none} GiB"
[ -n "$PEAK" ] && awk "BEGIN{exit !($PEAK < 15.0)}" || { log "8B SMOKE FAILED, stop"; exit 1; }
rm -rf $T/adapter_8b_smoke; gpu_free
mkdir -p $T/adapter_8b && cp $T/freeze_v41.sha256 $T/dataset_v41.sha256 $T/adapter_8b/
log "8B train start"
DATA_PATH=$D/dataset_v41_8b.jsonl OUT_DIR=$T/adapter_8b ~/train-venv/bin/accelerate launch --config_file ~/lora_pe_src/ddp2.yaml train_v4_8b_pack.py 2 > $T/train_8b.log 2>&1
rc=$?; log "8B train exit $rc"
[ $rc = 0 ] && [ -d $T/adapter_8b/final ] || { log "8B FAILED, 4B not started"; exit 1; }
gpu_free || { log "GPU NOT FREE after 8B, stop"; exit 1; }

# 6 4B(底模 V6):冒煙 → 全量
log "4B smoke"
BASE=$HOME/zeng_v6 MAX_STEPS=5 LOG_STEPS=1 SAVE_STEPS=100000 DATA_PATH=$D/dataset_v41_4b.jsonl OUT_DIR=$T/adapter_4b_smoke \
  ~/train-venv/bin/accelerate launch --config_file ~/lora_pe_src/ddp2.yaml train_v4_4b_pack.py 2 > $T/train_4b_smoke.log 2>&1
PEAK=$(grep -o "保留 [0-9.]* GiB" $T/train_4b_smoke.log | awk '{print $2}' | sort -n | tail -1)
log "4B smoke peak reserved ${PEAK:-none} GiB"
[ -n "$PEAK" ] && awk "BEGIN{exit !($PEAK < 15.0)}" || { log "4B SMOKE FAILED, stop"; exit 1; }
rm -rf $T/adapter_4b_smoke; gpu_free
mkdir -p $T/adapter_4b && cp $T/freeze_v41.sha256 $T/dataset_v41.sha256 $T/adapter_4b/
log "4B train start"
BASE=$HOME/zeng_v6 DATA_PATH=$D/dataset_v41_4b.jsonl OUT_DIR=$T/adapter_4b ~/train-venv/bin/accelerate launch --config_file ~/lora_pe_src/ddp2.yaml train_v4_4b_pack.py 2 > $T/train_4b.log 2>&1
rc=$?; log "4B train exit $rc"
[ $rc = 0 ] && [ -d $T/adapter_4b/final ] && log "CHAIN DONE" || log "4B FAILED"
