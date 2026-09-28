#!/bin/bash
# CPU 版 ComfyUI(不碰 GPU)→ 節點輸入比對 → 關掉
cd ~/ComfyUI_lab && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$HOME/lab_extra_x86 setsid -f ~/nunchaku-venv/bin/python main.py --cpu --listen 127.0.0.1 --port 8190 --disable-auto-launch > ~/mdh/comfy_cpu.log 2>&1
for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:8190/object_info/MasterStylePrompt && break; sleep 3; done
~/nunchaku-venv/bin/python ~/teacher_v4/node_io_check.py 8190 150 ~/teacher_v4/07_teacher > ~/teacher_v4/node_io_check.log 2>&1
for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null)
  case "$c" in "/home/pottokao/nunchaku-venv/bin/python main.py --cpu --listen 127.0.0.1 --port 8190 "*) kill ${p#/proc/};; esac; done
echo END >> ~/teacher_v4/node_io_check.log
