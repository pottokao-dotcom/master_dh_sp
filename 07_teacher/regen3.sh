#!/bin/bash
set -u
T=~/teacher_v4/07_teacher; L=~/teacher_v4/regen_design.log
docker rm -f gemma26m >> $L 2>&1
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}'); [ "$u" -lt 1000 ] && break; sleep 2; done
echo "$(date '+%T') GPUs free (${u} MiB used total)" >> $L
docker run -d --name gemma26m --gpus all --ipc=host -p 8012:8000 -v /home/pottokao/gemma4_26b_huihui_nvfp4:/model vllm/vllm-openai:nightly-bh /model --host 0.0.0.0 --port 8000 --served-model-name gemma26 --tensor-parallel-size 2 --disable-custom-all-reduce --language-model-only --kv-cache-dtype fp8 --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 --max-num-seqs 8 --speculative-config '{"method":"gemma4_mtp","model":"/model/assistant","num_speculative_tokens":4}' >> $L 2>&1
ok=0; for i in $(seq 1 150); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8012/v1/models | grep -q 200 && ok=1 && break; sleep 5; done
[ $ok = 1 ] || { echo "$(date '+%T') GEMMA FAILED TO START" >> $L; exit 1; }
curl -s http://127.0.0.1:8012/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gemma26","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}' > /dev/null && echo "$(date '+%T') gemma up (answers)" >> $L
cd $T && ~/nunchaku-venv/bin/python driver.py > driver_regen.log 2>&1
# 驅動沒寫 DONE 就視為失敗
echo "$(date '+%T') regen done: $(tail -n1 driver_regen.log)" >> $L
