#!/usr/bin/env python
"""v4 題材補齊(Gemma @ moon):新卡合風格題、複合長題、中文翻譯。輸出 seeds_*.jsonl,可續跑。"""
import json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / "04_node")); import master_style as ms
URL = "http://localhost:8012/v1/chat/completions"
def ask(system, user, temp=1.0, maxt=3000):
    body = {"model": "gemma26", "temperature": temp, "top_p": 0.95, "top_k": 40, "max_tokens": maxt,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    raw = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                    {"Content-Type": "application/json"}), timeout=900))["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:                                   # 引號沒跳脫 → 逐行撈
        items = []
        for line in m.group(0).splitlines():
            line = line.strip().rstrip(",").strip()
            if len(line) > 2 and line[0] == '"' and line[-1] == '"':
                items.append(line[1:-1].replace('\\"', '"'))
        return items
SYS = ("You write short image requests the way ordinary users type them into an image generator. "
       "Plain, concrete, varied; no artist names, no style words, no camera terms. Output a JSON array of strings only.")

def new_card_seeds(code, n=60):
    c = ms.load_card(code)
    out = []
    for batch in range(n // 20):
        u = (f"Write 20 different image requests a user might type for pictures of subjects like these: "
             f"{ms.flat(c.get('habitual_subjects',''))}. Each 5 to 18 words, one concrete scene each: who or what, "
             f"doing what, where, when. About a third should include a short sign, title or label in double quotes "
             f"that must appear in the image. Vary them widely.")
        out += [s for s in ask(SYS, u) if isinstance(s, str)]
    return [{"seed_core": s, "card": code, "src": "gemma_new"} for s in out[:n]]

def composite_seeds(n=1500):
    kinds = ["concert poster", "film poster", "theatre poster", "travel poster", "exhibition poster", "album cover",
             "magazine cover", "book cover", "festival poster", "event flyer", "opera poster", "product advertisement"]
    out = []
    def one(k):
        u = (f"Write 10 different requests for a designed {k}. Each 20 to 40 words and contains BOTH: "
             f"(1) the lettering — a title plus one or two of: subtitle, date, venue, performer or tagline — each "
             f"in double quotes; (2) the picture on it — who or what, doing what, where, when. Mix English and "
             f"Chinese requests; Chinese requests use 「」 for the lettering and Traditional Chinese characters.")
        return [{"seed_core": s, "kind": k, "src": "gemma_composite"} for s in ask(SYS, u) if isinstance(s, str)]
    jobs = [kinds[i % len(kinds)] for i in range(n // 10)]
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, jobs): out += r
    return out

def translate(seeds):
    def one(chunk):
        u = ("Translate each image request into natural Traditional Chinese as a Taiwanese user would type it. "
             "Keep anything inside double quotes exactly as it is, but write the quotes as 「」. "
             "Return a JSON array with the same number of items, same order.\n" + json.dumps([s["seed_core"] for s in chunk], ensure_ascii=False))
        r = ask("You are a careful translator. Output a JSON array of strings only.", u, temp=0.3)
        return [dict(s, seed_core=t, lang="zh", src_en=s["seed_core"]) for s, t in zip(chunk, r) if isinstance(t, str)] if len(r) == len(chunk) else []
    chunks = [seeds[i:i + 20] for i in range(0, len(seeds), 20)]
    out = []
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, chunks): out += r
    return out

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "new":
        codes = [c for c in ms.list_cards() if ms.load_card(c).get("source") == "new" and ms.load_card(c).get("medium") != "design"]
        out = []
        with ThreadPoolExecutor(8) as ex:
            for r in ex.map(new_card_seeds, codes): out += r
        Path(H / "seeds_newcards.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
        print(len(codes), "cards", len(out), "seeds")
    elif what == "composite":
        out = composite_seeds()
        Path(H / "seeds_composite.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
        print(len(out), "composite seeds")
    elif what == "zh":
        src = [json.loads(l) for l in open(sys.argv[2])]
        random.seed(7); pick = random.sample(src, min(int(sys.argv[3]), len(src)))
        out = translate(pick)
        Path(H / "seeds_zh.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
        print(len(out), "translated")
