#!/usr/bin/env python
"""v4 老師資料驅動(在 moon 上 nohup 跑,可續跑):jobs.jsonl → Gemma(本機 :8012)→ 篩選 → accepted/failed。
學生樣本存「正常 SP」的 system/user + 學生格式答案;老師格式只用來產答案。規則見 DESIGN 第 13 節。"""
import json, re, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / "04_node")); import master_style as ms
URL = "http://127.0.0.1:8012/v1/chat/completions"
ACC, FAIL, PROG = H / "accepted.jsonl", H / "failed.jsonl", H / "PROGRESS.txt"
NAMES = re.compile(open(H / "names.regex", encoding="utf-8").read().strip(), re.I)
QUOTED = re.compile(r"「([^」]+)」|『([^』]+)』|\"([^\"]+)\"|“([^”]+)”")
AGE = re.compile(r"\b(early|mid|late)[- ]?(teens|twenties|thirties|forties|fifties|sixties|seventies|eighties)\b|\b\d{1,2}[- ]?(year|yr)s?[- ]old\b", re.I)
LAMP = re.compile(r"\b(lamp|bulb|lantern|flash unit|strobe|spotlight|light fixture|flashgun)s?\b", re.I)
MOOD = re.compile(r"\b(the )?(overall )?mood (is|of)\b", re.I)
NUDE = re.compile(r"\b(nude|naked|topless|unclothed)\b|wears? nothing|nothing but (a |her |his |their )?(pair of )?[\w-]*\s?(heels|shoes|stockings|jewell?ery)|bare (breasts?|chest|buttocks)", re.I)
KID = re.compile(r"\b(child|children|boy|girl|kid|teen\w*|schoolgirl|schoolboy)\b|孩|童|少女|少年", re.I)
SEX = re.compile(r"\b(sensual|seductive|provocative|coquett\w*|flirt\w*|alluring|sultry|erotic|suggestive|lingerie)\b", re.I)
PAINT_W = re.compile(r"\b(painting|painted|oil|watercolou?r|pastel|woodblock|engraving|ink wash|drawing|gouache|impasto|brushstroke|brushwork|on silk|pointillis|canvas|glaze|fresco|tempera|gongbi|lithograph)\w*", re.I)
PHOTO_W = re.compile(r"\b(photograph\w*|photo|photomontage|snapshot|film still)\b", re.I)
DESIGN_KW = {k: re.compile(v, re.I) for k, v in json.load(open(H / "design_kw.json")).items()}
MINLEN = {"zimage": 170, "qwen_image": 280, "composite": 380, "composite_zimage": 230}
STOP = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "to", "by", "for", "his", "her", "their", "its"}
lock = threading.Lock()
squash = lambda t: re.sub(r"[\s\"“”「」『』/|\\\\,.，。]+", "", t).lower()

def ask(system, user):
    body = {"model": "gemma26", "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
            "max_tokens": 5000, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for k in range(5):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                             {"Content-Type": "application/json"}), timeout=900))["choices"][0]["message"]["content"]
        except Exception as e:
            time.sleep(10 * (k + 1)); err = e
    raise err

def text_checks(t):
    """v4.1b:引號外的描述寫比例 / 引號壞 / 後設話(組資料時也套到 v4 無版面資料)。"""
    why = []
    body = re.sub(r'"[^"]*"', "", t)
    if re.search(r"\b\d{1,2}:\d{1,2}\b|aspect ratio", body, re.I): why.append("ratio_in_text")
    if re.search(r'""\w|\w""', t): why.append("broken_quote")
    if re.search(r"\(wait\b|\bwait,|in the task\b|the task (asks|says|requires|specifies|names|calls)|as requested|"
                 r"per the (task|rules)|omitted per|the user\b|\bprompt\b", t, re.I): why.append("meta")
    return why

def ratio_ignored(job, d):
    """v4.1c:使用者用旋鈕指定比例,老師卻換成別的(v4 無版面 109/2237、v4.1b 複合 61/345)。"""
    r = (job.get("knobs") or {}).get("ratio", "auto")
    return r != "auto" and str(d.get("wh_ratio", "")) != r

def check(job, engine, d, prompt):
    """回傳不合格理由(空 = 通過)。"""
    why = []
    t = d["rewritten_prompt"]; tl = t.lower()
    kind = (("composite_zimage" if engine == "zimage" else "composite") if job["design"] != "none" else engine)
    if len(t.split()) < MINLEN[kind]: why.append("short")
    for q in QUOTED.findall(prompt):
        q = next(x for x in q if x)
        if squash(q) not in squash(t): why.append("quoted:" + q[:20])
    for it in d.get("task_items", []):
        w = [x for x in re.findall(r"[\w'-]+", str(it).lower()) if x not in STOP]
        if not w or " ".join(w) in tl: continue
        long = [x for x in w if len(x) >= 4] or w
        if sum(x in tl for x in long) < max(1, len(long) / 2): why.append("item:" + str(it)[:20])
    if AGE.search(t): why.append("age")
    if engine == "zimage" and LAMP.search(t) and not LAMP.search(prompt): why.append("lamp")
    if MOOD.search(" ".join(re.split(r"(?<=[.!?])\s+", t.strip())[:-1])): why.append("mood")
    if "master_style" in t or NAMES.search(t): why.append("name_or_tag")
    if not ms.NUDE_ASKED.search(prompt) and NUDE.search(t): why.append("nude_not_asked")
    if KID.search(prompt) and SEX.search(t): why.append("kid_sexual")
    if not re.fullmatch(r"\d+:\d+", str(d.get("wh_ratio", ""))): why.append("ratio")
    if len(d.get("task_items", [])) > 8: why.append("items>8")
    why += text_checks(t)
    if ratio_ignored(job, d): why.append("user_ratio")
    if job["design"] != "none":                           # v4.1:複合的媒材反轉(畫家卡開頭寫成照片、攝影卡開頭寫成畫)
        med = ms.medium_of(job["card"]) if job["card"] not in ("none", ms.BASE) else "none"
        first = re.split(r"(?<=[.!?])\s+", t.strip())[0]
        if med == "painting" and (PHOTO_W.search(first) or not PAINT_W.search(t)): why.append("medium:photo")
        if med == "photo" and PAINT_W.search(first) and not PHOTO_W.search(first): why.append("medium:paint")
    if job["design"] != "none" and len({m.group(0).lower() for m in DESIGN_KW[job["design"]].finditer(t)}) < 2:
        why.append("design_ignored")                      # 版面卡的招牌特徵沒寫出來(抽檢 3/41 被 Gemma 忽略)
    return why

def run(job, engine):
    k = dict(job["knobs"]); seed = k.pop("seed", 0)
    kw = dict(mood=k.get("mood", "auto"), framing=k.get("framing", "auto"), strength=k.get("strength", "standard"),
              habitual=k.get("habitual", False), preset=k.get("preset", "none"), ratio=k.get("ratio", "auto"), seed=seed)
    st_s, st_u, neg, asp = ms.build(job["prompt"], job["card"], engine=engine, fmt="roleplay2", design=job["design"], **kw)
    te_s, te_u, _, _ = ms.build_teacher(job["prompt"], job["card"], engine=engine, design=job["design"], **kw)
    reasons = []
    for attempt in range(1, 4):
        raw = ask(te_s, te_u)
        try:
            d = ms.join_teacher(json.loads(re.search(r"\{.*\}", raw, re.S).group(0)))
        except Exception:
            reasons.append("json"); continue
        why = check(job, engine, d, job["prompt"])
        if not why:
            rec = {"id": job["id"], "engine": engine, "card": job["card"], "design": job["design"], "kind": job["kind"],
                   "knobs": job["knobs"], "prompt": job["prompt"], "system": st_s, "user": st_u, "negative": neg,
                   "aspect": asp, "target": json.dumps(d, ensure_ascii=False), "attempts": attempt, "rejected": reasons}
            with lock: open(ACC, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            return True
        reasons.append("|".join(why))
    with lock: open(FAIL, "a").write(json.dumps({"id": job["id"], "engine": engine, "card": job["card"],
                                                  "kind": job["kind"], "prompt": job["prompt"], "reasons": reasons}, ensure_ascii=False) + "\n")
    return False

if __name__ == "__main__":
    jobs = [json.loads(l) for l in open(H / "jobs_v41_composite.jsonl")]
    if len(sys.argv) > 1: jobs = jobs[:int(sys.argv[1])]
    done = set()
    for f in (ACC, FAIL):
        if f.exists():
            done |= {(json.loads(l)["id"], json.loads(l)["engine"]) for l in open(f)}
    todo = [(j, e) for j in jobs for e in j["engines"] if (j["id"], e) not in done]
    total, t0, ok, n = len(todo), time.time(), 0, 0
    def one(a):
        global ok, n
        r = run(*a)
        with lock:
            n += 1; ok += r
            if n % 20 == 0 or n == total:
                el = time.time() - t0
                PROG.write_text(f"{time.strftime('%F %T')} done {n}/{total} accepted {ok} ({ok/n:.1%}) "
                                f"{n/el*60:.1f}/min ETA {(total-n)/(n/el)/3600:.2f} h\n")
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(one, todo))
    print("DONE", ok, "/", total)
