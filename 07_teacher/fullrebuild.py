"""全量重組:accepted.jsonl 每一筆用凍結版程式重組 system/user/negative,必須 100% 一樣(證明是同一版程式產的)。"""
import json,sys
sys.argv=['x']; sys.path.insert(0,'/home/pottokao/teacher_v4/07_teacher')
import audit
from collections import Counter
c=Counter()
for l in open('/home/pottokao/teacher_v4/07_teacher/accepted.jsonl'):
    r=json.loads(l); c[('有版面' if r['design']!='none' else '無版面', audit.rebuild(r))]+=1
print('全量重組', dict(c), flush=True)
sys.exit(0 if all(ok for (_,ok) in c) else 1)
