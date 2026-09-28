#!/bin/bash
# 單一順序鏈:同一時間 moon 只跑一件 GPU 工作(不再互搶)
set -u
T=~/teacher_v4
mv $T/regen_design.log $T/regen_design.crash1404.log 2>/dev/null
echo "$(date '+%T') 1/4 出污染圖 A舊老師 + C學生v4" > $T/serial.log
bash $T/contam_render.sh
echo "$(date '+%T') 2/4 Gemma 重生版面資料" >> $T/serial.log
bash $T/regen3.sh
echo "$(date '+%T') 3/4 稽核→資料集→B新老師出圖→8B" >> $T/serial.log
bash $T/after_regen.sh
echo "$(date '+%T') 4/4 chain exit $?" >> $T/serial.log
