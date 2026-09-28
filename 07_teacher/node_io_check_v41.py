"""真 ComfyUI 節點驗輸入:MasterStylePrompt 吐出的 system/user 逐字比對訓練料 messages[0]/[1]。
只跑 MS 節點 + PreviewAny,ComfyUI 用 CPU(不碰 GPU)。用法: node_io_check.py <port> <n_each> <dataset_dir>"""
import json, sys, random, time, urllib.request
sys.path.insert(0, '/home/pottokao/mdh/ComfyUI-MasterStyle/core'); import master_style as ms
PORT, N, D = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
C = f"http://127.0.0.1:{PORT}"
jobs = {j['id']: j for j in map(json.loads, open('/home/pottokao/teacher_v4/07_teacher/jobs.jsonl'))}
lab = lambda c: ms.load_card(c)['label_zh'] if c != 'none' else 'none'
def post(g):
    r = json.load(urllib.request.urlopen(urllib.request.Request(C + "/prompt", json.dumps({"prompt": g}).encode(), {"Content-Type": "application/json"})))
    pid = r["prompt_id"]
    while True:
        h = json.load(urllib.request.urlopen(f"{C}/history/{pid}"))
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error": raise RuntimeError(json.dumps(st)[:500])
            return h[pid]["outputs"]
        time.sleep(0.2)
random.seed(7); stats = {}
for fn, eng in (("dataset_v41_4b.jsonl", "zimage"), ("dataset_v41_8b.jsonl", "qwen_image")):
    rows = [json.loads(l) for l in open(f"{D}/{fn}")]
    des = [r for r in rows if r.get('design', 'none') != 'none']; plain = [r for r in rows if r.get('design', 'none') == 'none']
    pick = random.sample(des, min(N // 2, len(des))) + random.sample(plain, N - min(N // 2, len(des)))
    ok = bad = 0
    for r in pick:
        j = jobs[r['id'].split('_')[0]]; k = j['knobs']
        g = {"MS": {"class_type": "MasterStylePrompt", "inputs": {"prompt": j["prompt"], "style": lab(j['card']), "layout": lab(j['design']),
               "engine": eng, "mood": k.get("mood", "auto"), "framing": k.get("framing", "auto"), "palette": k.get("palette", "auto"),
               "strength": k.get("strength", "standard"), "habitual_subjects": k.get("habitual", False), "preset": k.get("preset", "none"),
               "length": "engine", "ratio": k.get("ratio", "auto"), "seed": k.get("seed", 0)}},
             "S": {"class_type": "PreviewAny", "inputs": {"source": ["MS", 0]}}, "U": {"class_type": "PreviewAny", "inputs": {"source": ["MS", 1]}}}
        o = post(g); s, u = o["S"]["text"][0], o["U"]["text"][0]
        m = r['messages']
        if s == m[0]['content'] and u == m[1]['content']: ok += 1
        else:
            bad += 1
            if bad <= 3:
                import difflib
                for a, b, nm in ((s, m[0]['content'], 'system'), (u, m[1]['content'], 'user')):
                    if a != b: print(r['id'], nm, 'DIFF:', ''.join(difflib.unified_diff(b.splitlines(1), a.splitlines(1), 'train', 'node', n=0))[:800])
    kinds = {}
    for r in pick: kinds[(r.get('design', 'none') != 'none', ms.load_card(r['card']).get('medium', 'photo'))] = kinds.get((r.get('design', 'none') != 'none', ms.load_card(r['card']).get('medium', 'photo')), 0) + 1
    print(fn, eng, "一樣", ok, "/", ok + bad, "| 抽到(有版面,媒材):", kinds, flush=True)
