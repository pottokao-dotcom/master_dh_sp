#!/usr/bin/env python
"""顯存壓艙:在大卡上佔住多餘顯存,模擬小卡(8G / 12G)。其他程式(ComfyUI、llama.cpp)只看得到剩下的量。

用法(先開這個,等它印 READY,再開 ComfyUI):
    CUDA_VISIBLE_DEVICES=0 python vram_ballast.py --free 7.6     # 模擬 8G 卡(扣掉驅動與桌面,實際可用約 7.6 GiB)
    CUDA_VISIBLE_DEVICES=0 python vram_ballast.py --free 11.5    # 模擬 12G 卡
Ctrl-C 或 kill 這個 PID 就釋放。只模擬容量,不模擬小卡較慢的頻寬與 PCIe。
"""
import argparse
import signal
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--free", type=float, required=True, help="要留給其他程式的顯存(GiB)")
ap.add_argument("--chunk", type=int, default=64, help="每次佔用的塊大小(MiB)")
a = ap.parse_args()

GiB, MiB = 1 << 30, 1 << 20
target = int(a.free * GiB)
torch.cuda.init()
free, total = torch.cuda.mem_get_info()
if free < target:
    raise SystemExit(f"目前只剩 {free / GiB:.2f} GiB,比要留的 {a.free} GiB 還少")
held = []
while True:                                       # 小塊逐步佔,直到剩下的剛好是目標
    free, _ = torch.cuda.mem_get_info()
    need = free - target
    if need <= 0:
        break
    held.append(torch.empty(min(need, a.chunk * MiB), dtype=torch.uint8, device="cuda"))
free, _ = torch.cuda.mem_get_info()
print(f"READY 卡 {total / GiB:.2f} GiB · 壓艙佔 {sum(t.numel() for t in held) / GiB:.2f} GiB · "
      f"其他程式可用 {free / GiB:.2f} GiB", flush=True)
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("釋放", flush=True)
