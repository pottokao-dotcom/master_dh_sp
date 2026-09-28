"""ComfyUI-MasterStyle:攝影風格卡 → 雙頭龍擴寫頭的 system/user,以及擴寫後的收尾(保險＋招牌句)。
接法:Master Style Prompt 的 system → Dual-Head Rewrite 的 system_prompt_text,user → prompt;
Rewrite 的 rewritten_prompt → Master Style Finalize → DiT 的 text-encode。設計見 DESIGN.md(不隨包出貨)。"""
import json
import os
import re
import sys

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "core"))
import master_style as ms  # noqa: E402

T2S = json.load(open(os.path.join(HERE, "t2s_chars.json"), encoding="utf-8"))
QUOTED = re.compile(r"「([^」]+)」|『([^』]+)』|\"([^\"]+)\"|“([^”]+)”")


def _styles():
    """兩個下拉:圖的風格(照片/繪畫卡,PE 預設卡第一個)、版面風格(design 卡,none 第一個)。"""
    cards = ms.list_cards()
    img = {ms.load_card(ms.BASE)["label_zh"]: ms.BASE}
    lay = {"none": "none"}
    for code in sorted(cards):
        if code == ms.BASE:
            continue
        c = ms.load_card(code)
        (lay if c.get("medium") == "design" else img)[c["label_zh"]] = code
    return img, lay


STYLES, LAYOUTS = _styles()


def guard_quotes(prompt, text):
    """保險 1:引號裡的原字若在輸出裡被寫成簡體,換回原字(單字對照表,來源 OpenCC,Apache-2.0)。"""
    for q in QUOTED.findall(prompt or ""):
        q = next(x for x in q if x)
        s = "".join(T2S.get(ch, ch) for ch in q)
        if q not in text and s != q and s in text:
            text = text.replace(s, q)
    return text


def guard_garbled(text):
    """保險 2(節點內部分):刪掉 U+FFFD。換 seed 重生要在擴寫節點裡做,併入雙頭龍時補。"""
    if "�" in text:
        print("[MasterStyle] rewrite contained U+FFFD; removed. Re-run with another seed for a clean result.")
        text = re.sub(r"\s*�+\s*", " ", text).strip()
    return text


class MasterStylePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MasterStylePrompt",
            display_name="Master Style Prompt",
            category="DualHeadDragon",
            description="Builds the role-play system prompt (style card at the end) and the user message for "
                        "Dual-Head Rewrite. Connect system → system_prompt_text and user → prompt.",
            inputs=[
                io.String.Input("prompt", multiline=True),
                io.Combo.Input("style", options=list(STYLES), default=list(STYLES)[0]),
                io.Combo.Input("layout", options=list(LAYOUTS), default="none",
                               tooltip="A poster/cover layout around the picture. Best with qwen_image (lettering)."),
                io.Combo.Input("engine", options=list(ms.SP_OF_ENGINE), default="qwen_image",
                               tooltip="qwen_image uses the long system prompt; zimage and klein use the short one."),
                io.Combo.Input("mood", options=list(ms.MOODS), default="auto"),
                io.Combo.Input("framing", options=list(ms.FRAMINGS), default="auto"),
                io.Combo.Input("palette", options=list(ms.PALETTES), default="auto",
                               tooltip="auto: the style's own palette; a colour named in the prompt always wins."),
                io.Combo.Input("strength", options=list(ms.STRENGTHS), default="standard"),
                io.Boolean.Input("habitual_subjects", default=False,
                                 tooltip="Let the style's usual subjects appear where the scene has room for them."),
                io.String.Input("preset", default="none", tooltip="A preset key of the chosen style, or none."),
                io.Combo.Input("length", options=list(ms.LENGTHS), default="engine"),
                io.String.Input("ratio", default="auto"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             tooltip="Only picks colour or black-and-white for styles that allow both."),
            ],
            outputs=[
                io.String.Output(display_name="system"),
                io.String.Output(display_name="user"),
                io.String.Output(display_name="negative"),
                io.String.Output(display_name="native_ratio"),
            ],
        )

    @classmethod
    def execute(cls, prompt, style, layout, engine, mood, framing, palette, strength, habitual_subjects, preset,
                length, ratio, seed) -> io.NodeOutput:
        code = STYLES.get(style, ms.BASE)
        s, u, n, r = ms.build(prompt, code, engine=engine, mood=mood, framing=framing, palette=palette,
                              strength=strength, habitual=habitual_subjects, preset=preset or "none",
                              length=length, ratio=ratio, seed=seed, fmt="roleplay2",
                              design=LAYOUTS.get(layout, "none"))
        return io.NodeOutput(s, u, n, r)


def ratio_size(ratio, fallback, area=1024 * 1024):
    """擴寫給的比例 → 寬高(約 area 像素、對齊 16);比例壞掉就用風格卡的原生比例,再不行 1:1。"""
    import math
    for r in (ratio, fallback, "1:1"):
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x×]\s*(\d+(?:\.\d+)?)\s*", r or "")
        if m and float(m.group(1)) > 0 and float(m.group(2)) > 0:
            k = float(m.group(1)) / float(m.group(2))
            w = math.sqrt(area * k)
            return int(round(w / 16) * 16), int(round(w / k / 16) * 16)
    return 1024, 1024


class MasterStyleFinalize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MasterStyleFinalize",
            display_name="Master Style Finalize",
            category="DualHeadDragon",
            description="Repairs the rewrite (quoted words back to the user's script, garbled characters removed) "
                        "and wraps it in the style's opening and closing lines.",
            inputs=[
                io.String.Input("rewritten_prompt", multiline=True, force_input=True),
                io.String.Input("prompt", multiline=True, tooltip="The user's original prompt (same as Master Style Prompt)."),
                io.Combo.Input("style", options=list(STYLES), default=list(STYLES)[0]),
                io.Combo.Input("layout", options=list(LAYOUTS), default="none"),
                io.String.Input("wh_ratio", default="", optional=True, force_input=True,
                                tooltip="Connect Dual-Head Rewrite's wh_ratio; width/height follow it (about 1 MP)."),
                io.Int.Input("megapixels_x100", default=100, min=25, max=400, step=5, optional=True,
                             tooltip="Output area: 100 = about 1 megapixel."),
            ],
            outputs=[io.String.Output(display_name="final_prompt"), io.Int.Output(display_name="width"),
                     io.Int.Output(display_name="height")],
        )

    @classmethod
    def execute(cls, rewritten_prompt, prompt, style, layout="none", wh_ratio="", megapixels_x100=100) -> io.NodeOutput:
        code, design = STYLES.get(style, ms.BASE), LAYOUTS.get(layout, "none")
        text = guard_garbled(guard_quotes(prompt, rewritten_prompt or ""))
        native = ms.load_card(design if design != "none" else code).get("aspect", "") or ""
        w, h = ratio_size(wh_ratio, native, area=1024 * 1024 * megapixels_x100 / 100)
        return io.NodeOutput(ms.finalize(text, code, prompt, design), w, h)


class MasterStyleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MasterStylePrompt, MasterStyleFinalize]


async def comfy_entrypoint() -> MasterStyleExtension:
    return MasterStyleExtension()
