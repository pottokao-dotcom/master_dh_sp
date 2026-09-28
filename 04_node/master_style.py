#!/usr/bin/env python
"""大師風格 → 雙頭龍擴寫的真實 prompt。

輸入:風格卡(02_cards/*.yaml)+ 旋鈕 + 使用者 prompt + 引擎 type
輸出:(system, user, negative, default_ratio)

- system:依引擎挑長短版(8B=long、4B=short),固定不變,好做 prefix cache。
- 繼承:pe-default 卡當底,選到的大師卡有寫的欄位蓋過去,沒寫的沿用 PE。
  開關(palette / density / staging / exceptions)在這裡解析成寫好的一行,模型只看到結果。
- user 訊息:Request → Style → Settings → Engine → Request 再講一次(小模型對結尾最敏感)。
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = {"long": ROOT / "01_system/pe_master_long.txt",
          "short": ROOT / "01_system/pe_master_short.txt",
          "roleplay_short": ROOT / "01_system/pe_master_roleplay_short.txt",
          "roleplay_long": ROOT / "01_system/pe_master_roleplay_long.txt",
          "roleplay_multi": ROOT / "01_system/pe_master_roleplay_long.multi.txt",   # 繪畫/版面卡(長版,QI2.1)
          "roleplay_multi_short": ROOT / "01_system/pe_master_roleplay_short.multi.txt",
          "roleplay_design_short": ROOT / "01_system/pe_master_roleplay_short.design.txt",
          "roleplay_design": ROOT / "01_system/pe_master_roleplay_long.design.txt",
          "roleplay_design_nar": ROOT / "01_system/pe_master_roleplay_long.design_nar.txt",        # v4.1 B 組:「你是畫家…接下來你成為設計師」
          "roleplay_design_short_nar": ROOT / "01_system/pe_master_roleplay_short.design_nar.txt"}
COMPOSITE_MODE = os.environ.get("MS_COMPOSITE", "narrative")   # narrative = 角色扮演(預設,使用者定) / turns = A 組說明書口吻(只留作對照)
PIECE_RATIO = {"album cover": "1:1", "magazine cover": "3:4", "book cover": "2:3", "flyer": "2:3", "label": "1:1", "poster": "2:3"}
PIECE_KINDS = [(r"album|唱片|record|黑膠", "album cover"), (r"magazine|雜誌", "magazine cover"), (r"book|書", "book cover"),
               (r"flyer|傳單", "flyer"), (r"label|標籤", "label"), (r"poster|海報", "poster")]


OVERLAP = {   # 字能不能壓在圖上(使用者 2026-09-27:雜誌那種文圖交雜要能做,畫作不會 → 依媒材,不寫死)
    "photo": "Your colours and marks belong to the page around it. Your lettering may sit beside the photograph or run "
             "across it, as a {kind} of your school would — a masthead tucked behind the subject's head, cover lines "
             "laid over the image; under the type the photograph stays a photograph.",
    "painting": "The painting stays whole: your colours, marks and lettering belong to the page around it, never on "
                "the painted surface.",
    "illustration": "Your lettering may sit beside the picture or flow into it, as your school does.",
}


def piece_kind(prompt, design_code):
    for pat, kind in PIECE_KINDS:
        if re.search(pat, prompt or "", re.I):
            return kind
    return "album cover" if "album" in design_code else "poster"   # v4.1 複合兩棒(圖→版面)   # 版面/複合(短版,4B)   # 繪畫卡(短版,4B;複合不走 4B)
COMPOSITE_LENGTH_SHORT = ("350 to 400 words: most of it on the picture inside, then the layout and the lettering.")   # v4.1:字不再佔三分之一
COMPOSITE_LENGTH = ("550 to 650 words: most of it on the picture inside, then the layout and the lettering.")
SP_OF_ENGINE = {"qwen_image": "long", "zimage": "short", "klein": "short"}
CARDS = ROOT / "02_cards"
ENGINES = ROOT / "03_engines/engines.yaml"
BASE = "pe-default"
TAGS = ROOT / "01_system/style_tags.json"

MOODS = ("auto", "off", "weak", "medium", "strong")
FRAMINGS = ("auto", "close", "neutral", "wide")
PALETTES = ("auto", "colour", "monochrome", "saturated")
STRENGTHS = ("light", "standard", "full")
LENGTHS = ("engine", "short", "long")

PALETTE_TEXT = {
    "monochrome": "Black and white: describe everything in tones — dense black, charcoal, mid grey, "
                  "silvery highlight, chalky white — for clothing, skin, sky and light alike, and describe "
                  "light by its direction and hardness rather than its warmth. The opening sentence names the "
                  "subject first and then calls the photograph black and white.",
    "saturated": "strong, unashamed colour named with intense modifiers — hot, glossy, acid, "
                 "blood-red, candy pink — never muted, dusty or tasteful.",
}
DENSITY_TEXT = {
    "sparse": "sparse: few elements; describe the gradations of the empty field and the surface of "
              "the subject; never invent things to fill space.",
}
STAGING_TEXT = {
    "surreal": "the arrangement may be absurd — bodies stacked, an object at the wrong scale — but "
               "it is built from the user's subject and what is already in the scene.",
}
EXCEPTION_TEXT = {
    "blur": "motion blur, camera shake and soft focus are described as the image shows them",
    "tilt": "a tilted horizon stays tilted",
    "obscured": "what is seen through glass, rain, fog or reflection is described as partly lost",
}
STRENGTH_TEXT = {
    "light": "light — only the style's light, colour and texture carry over; everything else is a "
             "plain photograph of the user's scene",
    "standard": "standard — apply the whole style",
    "full": "full — the style decides every open choice in every sentence",
}
LENGTH_TEXT = {"short": "about half the engine's length", "long": "the top of the engine's range"}


def style_tag(code):
    """風格 tag(DESIGN 第 10 節):固定編號、永不重用,只放 system,絕不出現在輸出。"""
    code = BASE if code in (None, "", "none") else code
    return next((t["tag"] for t in json.loads(TAGS.read_text(encoding="utf-8")) if t["code"] == code), "")


def list_cards():
    return {p.stem: p for p in sorted(CARDS.glob("*.yaml"))}


def load_card(code):
    return yaml.safe_load(list_cards()[code].read_text(encoding="utf-8"))


def flat(v):
    if isinstance(v, list):
        v = "; ".join(str(x).strip() for x in v)
    return " ".join(str(v).split())


def merged(code):
    """PE 預設卡當底,大師卡有寫的欄位整欄蓋過去(framing / mood 也是整組換,不混兩張卡的檔位)。"""
    base = load_card(BASE)
    if code in (None, "", "none", BASE):
        return dict(base), False
    out = dict(base)
    out.update({k: v for k, v in load_card(code).items() if v not in (None, "", [], {})})
    return out, True


def style_block(c, is_master, mood, framing, palette, strength, habitual, preset, seed):
    """回傳 (look 區塊, write 區塊, settings 列)。
    look = 畫面描述,可以直接寫進 prompt;write = 寫作規則,不能抄進 prompt。"""
    rng = random.Random(seed)
    pal = c.get("palette", "colour") if palette == "auto" else palette
    if pal == "either":                                   # 黑白彩色都有的卡,依 seed
        pal = rng.choice(["monochrome", "colour"])
    mono = pal == "monochrome"
    lk = []
    if is_master and c.get("look_paragraph"):          # v3 卡(CARD_RULES R4):一段描述,不分欄位
        para = flat(c["look_paragraph"])
        if not mono and c.get("palette") in ("monochrome", "either"):   # 使用者點名顏色 → 當次彩色
            para = strip_bw(para)
        lk.append(("Black and white. " if mono else "") + para)
        m = c.get("mood") or {}
        if mood != "off" and m:
            lvl = m.get("default", "medium") if mood == "auto" else mood
            text = m.get(lvl) or m.get(m.get("default", "medium"))
            if text:
                lk.append(f"The feeling carried by bodies, gaze and light: {flat(text)}.")
        f = c.get("framing") or {}
        pos = f.get("native", "neutral") if framing == "auto" else framing
        if pos in ("close", "wide") and f.get(pos):
            lk.append(f"Framing: {flat(f[pos])}.")
        if habitual and c.get("habitual_subjects"):
            lk.append(f"Where the scene has room for them: {flat(c['habitual_subjects'])}.")
        return _finish(c, is_master, lk, pal, strength, preset, rng)
    for key, label in (("opening", "First impression"), ("look", "Look"), ("camera", "Camera"),
                       ("light", "Light")):
        text = flat(c[key])
        if mono and key in ("look", "opening"):           # 4B 會照抄這兩行 → 黑白寫在這
            text = "black-and-white — " + text
        lk.append(f"{label}: {text}")
    lk.append("Colour: " + ("black and white, in tones from dense black to chalky white" if mono
                            else flat(c["colour"])))
    lk.append(f"Texture: {flat(c['texture'])}")
    if c.get("expression"):
        lk.append(f"Faces: {flat(c['expression'])}")
    if c.get("attitude"):
        lk.append(f"Attitude: {flat(c['attitude'])}")
    m = c.get("mood") or {}
    if mood != "off" and m:
        lvl = m.get("default", "medium") if mood == "auto" else mood
        text = m.get(lvl) or m.get(m.get("default", "medium"))
        if text:
            lk.append(f"Mood: {flat(text)}")
    f = c.get("framing") or {}
    pos = f.get("native", "neutral") if framing == "auto" else framing
    if pos in ("close", "wide") and f.get(pos):
        lk.append(f"Framing: {flat(f[pos])}")
    for rule in c.get("must") or []:
        lk.append(f"Always: {flat(rule)}")
    sig = flat(c["keep"]) if c.get("keep") else ""
    if mono:
        sig = (sig + ", " if sig else "") + "black and white"
    if sig:                                             # 舊格式卡:寫成描述句,不帶欄位名
        lk.append(f"Always visible: {sig}")
    if habitual and c.get("habitual_subjects"):
        lk.append(f"Habitual subjects: {flat(c['habitual_subjects'])}")

    return _finish(c, is_master, lk, pal, strength, preset, rng)


def _finish(c, is_master, lk, pal, strength, preset, rng):
    wr = [f"Colour naming: {PALETTE_TEXT[pal] if pal in PALETTE_TEXT else flat(c['colour_naming'])}"]
    dens = c.get("density")
    wr.append(f"Density: {DENSITY_TEXT.get(dens, flat(dens))}")
    if c.get("staging") in STAGING_TEXT:
        wr.append(f"Staging: {STAGING_TEXT[c['staging']]}")
    if c.get("adult_mood"):                             # 有情慾/挑逗氣氛的卡:只屬於成人
        wr.append("Adults only: any sensual, teasing or erotic charge belongs to adults alone; a child or "
                  "teenager in the picture is photographed plainly and innocently, fully clothed.")
    phys = flat(c["physics"])
    exc = [EXCEPTION_TEXT[e] for e in c.get("exceptions") or [] if e in EXCEPTION_TEXT]
    if exc:
        phys += " — except: " + "; ".join(exc)
    wr.append(f"Physics: {phys}")

    se = []
    if c.get("aspect"):
        se.append(f"native ratio: {c['aspect']}")
    if is_master:
        se.append(f"style strength: {STRENGTH_TEXT[strength]}")
    presets = c.get("presets") or {}
    if preset == "random" and presets:
        preset = rng.choice(sorted(presets))
    if preset in presets:
        se.append(f"this picture leans toward: {flat(presets[preset])}")
    return "\n".join(lk), "\n".join(wr), se


def engine_block(engine, length):
    e = yaml.safe_load(ENGINES.read_text(encoding="utf-8"))[engine]
    out = [f"Length: {flat(e['length'])}"]
    if length in LENGTH_TEXT:
        out.append(f"Length for this image: {LENGTH_TEXT[length]}")
    out.append(f"Elements: at most {flat(e['elements'])}")
    out += [f"- {flat(n)}" for n in e.get("notes") or []]
    return "\n".join(out)


NUDE_ASKED = re.compile(r"\b(nude|naked|topless|bare[- ]chested|unclothed|undressed|nudity)\b|裸|赤身|全裸|半裸", re.I)
COLOUR_WORDS = re.compile(
    r"\b(red|orange|yellow|green|blue|purple|violet|pink|magenta|cyan|teal|turquoise|golden|gold|brown|"
    r"crimson|scarlet|maroon|navy|beige|lilac|lavender|colou?rs?|colou?rful)\b(?! hour)"
    r"|[紅橙黃綠藍紫粉棕褐金青]色|彩色|紅(?![茶燒])|粉紅", re.I)
BW = (re.compile(r"black-and-white |black and white |monochrome ", re.I), re.compile(r",? (in )?black and white\.$", re.I))


def user_colour(prompt):
    """使用者點名顏色(優先順序第一):黑白卡當次改彩色。"""
    return bool(COLOUR_WORDS.search(prompt or ""))


def strip_bw(text):
    t = BW[0].sub("", text)
    return BW[1].sub(".", t)


QUOTED = re.compile(r'「([^」]+)」|『([^』]+)』|"([^"]+)"|“([^”]+)”')
LETTERING = re.compile(r'["「『“”]|poster|海報|cover|封面|album|唱片|magazine|雜誌|flyer|傳單|label|標籤|標題|sign', re.I)


FORMAT_NOTE = {
    "json": "## How the message is packed\n\nThe user message is one JSON object. `request` and "
            "`request_again` are what the user wants. `look` is description you may reuse in your own "
            "sentences. `rules`, `settings` and `engine` are instructions to you: follow them, and never "
            "let their wording appear in the description.",
    "xml": "## How the message is packed\n\nThe user message is split by tags. <request> and "
           "<request_again> are what the user wants. <look> is description you may reuse in your own "
           "sentences. <rules>, <settings> and <engine> are instructions to you: follow them, and never "
           "let their wording appear in the description.",
    "sysrules": "",
}


def build_parts(prompt, code="none", engine="qwen_image", mood="auto", framing="auto",
                palette="auto", strength="standard", habitual=False, preset="none",
                length="engine", ratio="auto", seed=0):
    """組好的各段(還沒決定接法)。"""
    c, is_master = merged(code)
    colour_override = palette == "auto" and user_colour(prompt) and c.get("palette") in ("monochrome", "either")
    if colour_override:                                 # 優先順序第一:使用者點名的顏色
        palette = "colour"
    look, write, settings = style_block(c, is_master, mood, framing, palette, strength,
                                        habitual, preset, seed)
    neg = c.get("negative", "")
    if not NUDE_ASKED.search(prompt or ""):              # 沒要求裸體 → 規則要求寫出衣著,negative 擋裸體
        write += ("\nClothing: the task does not ask for nudity, so every person is dressed — name the garments "
                  "that cover the body (a dress, a slip, a shirt, trousers, a robe), not only shoes or jewellery; "
                  "bare skin shows only where those clothes leave it.")
        # (2026-09-28 使用者：拿掉自動加的 negative "nude, naked, topless"；衣著仍由上面 Clothing 規則管)
    if colour_override:
        neg = ", ".join(x for x in (y.strip() for y in neg.split(",")) if x.lower() not in ("colour", "color"))
    if ratio != "auto":
        settings.append(f"ratio: {ratio}")
    again = prompt.strip()
    words = ["".join(w) for w in QUOTED.findall(prompt)]
    if words:
        again += "\nWords shown in the image, exactly as written: " + ", ".join(f'"{w}"' for w in words)
    if LETTERING.search(prompt):                        # 小模型最容易漏字 → 結尾再提醒
        again += ("\nThe requested words appear in the image exactly as written, in quotation marks, as "
                  "the main title; a poster, cover or label also carries the smaller lettering its kind has.")
    return {"request": prompt.strip(), "look": look, "rules": write, "settings": "\n".join(settings),
            "engine": engine_block(engine, length), "request_again": again,
            "negative": neg, "aspect": c.get("aspect", ""), "colour_override": colour_override}


def medium_of(code):
    return "photo" if code in (None, "", "none") else load_card(code).get("medium", "photo")


def build(prompt, code="none", engine="qwen_image", mood="auto", framing="auto",
          palette="auto", strength="standard", habitual=False, preset="none",
          length="engine", ratio="auto", seed=0, sp=None, fmt="md", design="none"):
    """fmt:md(markdown 標題分段)/ json / xml / sysrules(規則與引擎段接在 system 尾)。"""
    system = SYSTEM[sp or SP_OF_ENGINE[engine]].read_text(encoding="utf-8").strip()
    P = build_parts(prompt, code, engine, mood, framing, palette, strength, habitual, preset,
                    length, ratio, seed)
    if fmt in ("roleplay", "roleplay2"):                               # 角色扮演:卡=角色(system 尾),user=任務
        design = None if design in (None, "", "none") else design
        multi = bool(design) or medium_of(code) != "photo"               # 繪畫/版面 → 多媒材 SP(QI2.1 長版)
        short = (sp or SP_OF_ENGINE[engine]) == "short"
        if multi:
            key = ("roleplay_design_short" if design else "roleplay_multi_short") if short else \
                  ("roleplay_design" if design else "roleplay_multi")
            if design and COMPOSITE_MODE == "narrative":
                key += "_nar"
        else:
            key = "roleplay_short" if short else "roleplay_long"
        system = SYSTEM[key].read_text(encoding="utf-8").strip()
        tag = style_tag(code)
        noun = {"painting": "paintings", "design": "pieces"}.get(medium_of(code), "photographs")
        eng = P["engine"]
        neg, aspect = P["negative"], P["aspect"]
        if design:
            eng = re.sub(r"Length: [^\n]*", "Length: " + (COMPOSITE_LENGTH_SHORT if short else COMPOSITE_LENGTH),
                         eng, count=1)
            D = load_card(design)
            dtag = style_tag(design)
            dlook = flat(D["look_paragraph"])
        look = P["look"]
        if design and code in (None, "", "none", BASE) and D.get("picture_paragraph"):   # 純版面:圖照版面流派本來的樣子,不用寫實攝影師
            look, noun = flat(D["picture_paragraph"]), "pictures"
        elif design and D.get("page_paragraph"):          # v4.1 複合:設計師第二棒只看頁面,不重畫圖卡的圖
            dlook = flat(D["page_paragraph"])
        if design and COMPOSITE_MODE == "narrative":       # B 組:你是畫家/攝影師 → 接下來你成為設計師,把剛剛那張做成封面
            head = re.split(r"[:,]", flat(D.get("picture_paragraph", "")) or "photo")[0]   # 純版面:看 picture_paragraph 第一段是照片/畫/插畫
            med = medium_of(code) if code not in (None, "", "none", BASE) else \
                next((m for k, m in (("photo", "photo"), ("paint", "painting")) if k in head), "illustration")
            who, work, verb = {"painting": ("a painter", "painting", "paint"), "photo": ("a photographer", "photograph", "photograph"),
                               "illustration": ("an illustrator", "picture", "draw")}[med]
            done = {"painting": "painted", "photo": "taken", "illustration": "drawn"}[med]
            kind = piece_kind(prompt, design)
            # 複合比例由成品決定(使用者有給就照使用者;沒點名成品 = 海報 2:3,同 SP):圖卡的 native ratio 換成成品的。
            # v4.1b:比例只放規則區的 native ratio 行 —— 寫進設計師那句話,老師會抄進描述(v4.1 19%)
            piece_ratio = ratio if ratio != "auto" else PIECE_RATIO[kind]
            settings = "\n".join(x for x in P["settings"].split("\n") if not x.startswith("native ratio:"))
            if ratio == "auto":
                settings = "\n".join(x for x in (f"native ratio: {piece_ratio}", settings) if x)
            words = ["".join(w) for w in QUOTED.findall(prompt)]
            system += ("\n\n## First, you are " + who + (f" ({tag})" if tag else "") + f"\n\nYou are {who}. Your {work}s look like this:\n"
                       + look + f"\n\nYou {verb} exactly the picture the task asks for, this way, and finish it."
                       + "\n\n## Then you become the designer" + (f" ({dtag})" if dtag else "")
                       + f"\n\nNext you become a {D.get('school_en', 'graphic')} graphic designer. You take the {work} you have just "
                       f"{done} and make it into {'an' if kind[0] in 'aeiou' else 'a'} {kind}. Your page looks like this:\n" + dlook
                       + f"\nThe {work} goes onto the page as it is: you place it, crop it and frame it, but you never repaint it. "
                       + OVERLAP[med].format(kind=kind)
                       + (("\nThe words on the " + kind + ": " + ", ".join(f'"{w}"' for w in words) + ".") if words else "")
                       + "\n\n## Your rules for this picture\n\n" + P["rules"] + "\n" + eng
                       + ("\n" + settings if settings else ""))
        else:
          system += ("\n\n## Your character" + (f" ({tag})" if tag else "") + f"\n\nYour {noun} look like this:\n" + look
                   + ((f"\n\n## The design" + (f" ({dtag})" if dtag else "") + "\n\nThe designer's turn: the finished picture "
                       "above is placed on a page laid out like this:\n" + dlook) if design else "")
                   + "\n\n## Your rules for this picture\n\n" + P["rules"] + "\n" + eng
                   + ("\n" + P["settings"] if P["settings"] else ""))
        user = "Task: " + P["request_again"]
        if design:
            if COMPOSITE_MODE == "narrative":              # v4.1:多印字最多兩行小字(使用者 09-27 交我定)
                user = user.replace("also carries the smaller lettering its kind has.",
                                    "may add at most two small lines of the other lettering its kind has.")
            if not LETTERING.search(prompt):
                user += ("\nThis is a designed piece, a poster unless the task names another kind; it may carry at most "
                         "two small lines of the lettering its kind has." if COMPOSITE_MODE == "narrative" else
                         "\nThis is a designed piece, a poster unless the task names another kind; it carries the "
                         "lettering its kind has.")
            # 犯沖處理:圖卡的色彩只管版面裡那張圖 → 圖卡 negative 拿掉顏色詞,規則加範圍說明
            neg = ", ".join(x for x in (y.strip() for y in neg.split(","))
                            if x.lower() not in ("colour", "color", "black and white", "monochrome", "muted colour",
                                                 "saturated", "desaturated"))
            neg = ", ".join(x for x in (neg, D.get("negative", "")) if x)
            if COMPOSITE_MODE != "narrative":              # 角色扮演版已在設計師那段用第二人稱講過
              system = system.replace("\n\n## Your rules for this picture\n\n",
                                    "\n\n## Your rules for this picture\n\nScope: the image-maker's colour, tone and "
                                    "texture belong to the picture inside; the page around it keeps the design's own "
                                    "colours and type.\n", 1)
            aspect = D.get("aspect", aspect) if ratio == "auto" else aspect
            if COMPOSITE_MODE == "narrative":
                aspect = piece_ratio
        if fmt == "roleplay2":                          # 4B 會太入戲 → 任務在結尾再講一次
            verb = "Photograph" if not multi else "Make"
            user += (f"\n\n{verb} exactly this, keeping every part of it — the place, the people, "
                     "what they are doing: " + P["request"])
        return system, user, neg, aspect
    if fmt == "json":
        keys = ("request", "look", "rules", "settings", "engine", "request_again")
        user = json.dumps({k: P[k] for k in keys if P[k]}, ensure_ascii=False, indent=1)
        system += "\n\n" + FORMAT_NOTE["json"]
    elif fmt == "xml":
        keys = ("request", "look", "rules", "settings", "engine", "request_again")
        user = "\n\n".join(f"<{k}>\n{P[k]}\n</{k}>" for k in keys if P[k])
        system += "\n\n" + FORMAT_NOTE["xml"]
    elif fmt == "sysrules":
        system += ("\n\n## Rules for this image\n\n" + P["rules"] + "\n\n## Engine\n\n" + P["engine"]
                   + ("\n\n## Settings\n\n" + P["settings"] if P["settings"] else ""))
        user = ("## Request\n" + P["request"] + "\n\n## The look — description you may reuse\n"
                + P["look"] + "\n\n## Request, once more\n" + P["request_again"])
    else:
        parts = ["## Request\n" + P["request"],
                 "## The look — describe the image this way; these words may be used in the description\n" + P["look"],
                 "## How to write it — rules for you, never copied into the description\n" + P["rules"]]
        if P["settings"]:
            parts.append("## Settings\n" + P["settings"])
        parts.append("## Engine\n" + P["engine"])
        parts.append("## Request, once more — its subject, its action and its objects all stay\n" + P["request_again"])
        user = "\n\n".join(parts)
    return system, user, P["negative"], P["aspect"]


def signatures(code="none", prompt="", design="none"):
    """頭尾招牌句;使用者點名顏色時,黑白卡的招牌句拿掉黑白字;只有版面卡時用版面卡的;
    版面卡＋圖卡時,頭＝版面卡 signature_lead_composite(沒有就用 signature_lead),尾＝圖卡的。"""
    img = code not in (None, "", "none", BASE)
    if img:
        c = load_card(code)
        lead, close = flat(c.get("signature_lead", "") or ""), flat(c.get("signature_close", "") or "")
        if user_colour(prompt) and c.get("palette") in ("monochrome", "either"):
            lead, close = strip_bw(lead), strip_bw(close)
    if design not in (None, "", "none"):
        D = load_card(design)
        if not img:
            return flat(D.get("signature_lead", "") or ""), flat(D.get("signature_close", "") or "")
        # 複合(2026-09-27):開頭用版面卡的無媒材版,結尾留圖卡的(畫家/攝影的畫法不被版面卡蓋掉)
        return flat(D.get("signature_lead_composite") or D.get("signature_lead", "") or ""), close
    if not img:
        return "", ""
    return lead, close


def finalize(text, code="none", prompt="", design="none"):
    """節點端:把招牌句夾在擴寫頭尾(CARD_RULES R6)。沒有招牌句的卡原樣回傳。"""
    if code in (None, "", "none", BASE) and design in (None, "", "none"):
        return text
    lead, close = signatures(code, prompt, design)
    return " ".join(x for x in (flat(lead) if lead else "", text.strip(), flat(close) if close else "") if x)


# ── 老師專用(DESIGN 第 8 節):只在產老師資料時用,學生訓練與實際使用一律用正常 SP ──
TEACHER_PARTS = ["opening", "subject", "regions", "lettering", "light", "composition", "closing"]


TEACHER_PARTS_COMPOSITE = ["opening", "subject", "regions", "light", "composition", "layout", "lettering", "closing"]   # v4.1:先圖(第一棒)後頁面(第二棒)


def teacher_output_block_composite(engine="qwen_image"):
    c = yaml.safe_load(ENGINES.read_text(encoding="utf-8"))[engine]["teacher_parts_composite"]
    return f"""## Output

First settle the task: list what it names — the people, each object, the place, the time,
the action, any words to be shown — in English, translated literally, as short phrases. Only
what the task itself names, each item once, at most eight items. Then write the description
in parts, and use every listed item in it, by that same word. Each part continues the one
before it; together they read as one description of the whole designed piece. The picture's
parts are your first turn, written as the one who made the picture, in its own medium; the
layout and lettering are your second turn, as its designer.

One strictly valid JSON object on a single line, nothing else:

{{"task_items": ["<item>"], "opening": "<one sentence naming the piece and its picture, with the picture's own medium>", "subject": "<{c['subject']} sentences on the picture's subject>", "regions": "<{c['regions']} sentences on the rest of the picture, region by region>", "light": "<{c['light']} sentences on the picture's light>", "composition": "<{c['comp']} sentences on the picture's composition>", "layout": "<{c['layout']} sentences: where the picture sits on the page and how it is framed or cropped, the grid, colour fields, borders, margins>", "lettering": "<{c['lettering']} sentences: every requested word exactly, in straight double quotes, then at most two small lines of the other lettering the piece carries>", "closing": "<one sentence on the whole piece>", "wh_ratio": "<e.g. 2:3>"}}"""


def teacher_output_block(engine):
    c = yaml.safe_load(ENGINES.read_text(encoding="utf-8"))[engine]["teacher_parts"]
    return f"""## Output

First settle the task: list what it names — the people, each object, the place, the time,
the action, any words to be shown — in English, translated literally, as short phrases. Only
what the task itself names, each item once, at most eight items. Then write the description
in parts, and use every listed item in it, by that same word. Each part continues the one
before it; together they read as one description in the order above.

One strictly valid JSON object on a single line, nothing else:

{{"task_items": ["<item>"], "opening": "<one sentence>", "subject": "<{c['subject']} sentences>", "regions": "<{c['regions']} sentences, region by region out to the edges>", "lettering": "<only if the task shows words, else empty>", "light": "<{c['light']} sentences>", "composition": "<{c['comp']} sentences>", "closing": "<one sentence>", "wh_ratio": "<e.g. 3:2>"}}"""


def build_teacher(prompt, code="none", engine="qwen_image", **kw):
    """老師版 system:把 ## Output 換成分段 JSON、拿掉 Length 行(長度改由各段句數決定)。"""
    s, u, n, r = build(prompt, code, engine=engine, fmt="roleplay2", **kw)
    composite = kw.get("design") not in (None, "", "none")
    i, j = s.index("## Output"), s.index("## First, you are" if "## First, you are" in s else "## Your character")
    s = s[:i] + (teacher_output_block_composite(engine) if composite else teacher_output_block(engine)) + "\n\n" + s[j:]
    s = re.sub(r"Length: [^\n]*\n", "", s, count=1)
    return s, u, n, r


def join_teacher(d):
    """老師分段 JSON → 學生格式 {task_items, rewritten_prompt, wh_ratio}。"""
    parts = TEACHER_PARTS_COMPOSITE if "layout" in d else TEACHER_PARTS
    text = " ".join(d.get(p, "").strip() for p in parts if d.get(p, "").strip())
    return {"task_items": d.get("task_items", []), "rewritten_prompt": text, "wh_ratio": d.get("wh_ratio", "")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--code", default="none", choices=["none"] + list(list_cards()))
    ap.add_argument("--engine", default="qwen_image", choices=list(SP_OF_ENGINE))
    ap.add_argument("--mood", default="auto", choices=MOODS)
    ap.add_argument("--framing", default="auto", choices=FRAMINGS)
    ap.add_argument("--palette", default="auto", choices=PALETTES)
    ap.add_argument("--strength", default="standard", choices=STRENGTHS)
    ap.add_argument("--habitual", action="store_true")
    ap.add_argument("--preset", default="random")
    ap.add_argument("--length", default="engine", choices=LENGTHS)
    ap.add_argument("--ratio", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-system", action="store_true")
    ap.add_argument("--fmt", default="md", choices=["md", "json", "xml", "sysrules", "roleplay", "roleplay2"])
    a = ap.parse_args()
    s, u, n, r = build(a.prompt, a.code, a.engine, a.mood, a.framing, a.palette, a.strength,
                       a.habitual, a.preset, a.length, a.ratio, a.seed, fmt=a.fmt)
    if a.show_system:
        print(s, "\n\n" + "=" * 60)
    print(u)
    print("\n" + "=" * 60 + f"\nnegative: {n}\ndefault ratio: {r}")
