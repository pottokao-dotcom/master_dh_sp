"""v4.1 訓練料組合+全量重組:v4 無版面(design=none)＋v4.1 複合 → accepted_all.jsonl;每筆用 v4.1 凍結程式重組 system/user/negative 必須 100% 一樣。"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.expanduser('~/teacher_v41/04_node')); import master_style as ms
assert ms.COMPOSITE_MODE == 'narrative'
T = os.path.expanduser('~/teacher_v41/07_teacher/')
old = [json.loads(l) for l in open(os.path.expanduser('~/teacher_v4/07_teacher/accepted.jsonl'))]
plain = [r for r in old if r['design'] == 'none']
comp = [json.loads(l) for l in open(T + 'accepted.jsonl')]
sys.argv = ['x']; sys.path.insert(0, T); import driver_v41                 # v4.1b:同一套文字規則濾 v4 無版面(只丟不改)
def bad(r):
    d = json.loads(r['target'])
    return driver_v41.text_checks(d['rewritten_prompt']) + (["user_ratio"] if driver_v41.ratio_ignored(r, d) else [])
for name in ('plain', 'comp'):                    # v4.1b/c:同一套規則濾兩邊(只丟不改)
    rows = globals()[name]
    print(name, '濾掉', dict(Counter(w for r in rows for w in bad(r))), flush=True)
    globals()[name] = [r for r in rows if not bad(r)]
print('剩 無版面', len(plain), '複合', len(comp), flush=True)
assert all(r['design'] != 'none' for r in comp)
def rebuild(r):
    k = dict(r["knobs"]); seed = k.pop("seed", 0)
    s, u, n, a = ms.build(r["prompt"], r["card"], engine=r["engine"], fmt="roleplay2", design=r["design"], seed=seed,
                          mood=k.get("mood", "auto"), framing=k.get("framing", "auto"), strength=k.get("strength", "standard"),
                          habitual=k.get("habitual", False), preset=k.get("preset", "none"), ratio=k.get("ratio", "auto"))
    return s == r["system"] and u == r["user"] and n == r["negative"]
c = Counter()
for r in plain + comp: c[('複合' if r['design'] != 'none' else '無版面', rebuild(r))] += 1
print('v4 無版面', len(plain), '| v4.1 複合', len(comp), '| 全量重組', dict(c), flush=True)
if not all(ok for (_, ok) in c): sys.exit(1)
open(T + 'accepted_all.jsonl', 'w').write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in plain + comp))
