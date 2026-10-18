# -*- coding: utf-8 -*-
"""按 manifest.json 渲染几套完整搭配，验证叠放顺序与可用性。

渲染顺序（按实际图层构造推出）：
    素体 body -> shoes -> hair -> bottom -> top/dress -> acc
注意文档第十节写的 shoes(5) 低于 素体身体(10)，照那个顺序画鞋会被身体盖住；
素体应当是 z=0 的底，鞋在它之上。
"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = "/Users/zhujinhui/claudeProject5/dressup-game/res"
M = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
ORDER = ["shoes", "hair", "bottom", "top", "dress", "acc"]

LOOKS = [
    ("日常", {"hair": "hair_02", "top": "top_01", "bottom": "bottom_03",
             "shoes": "shoes_02", "acc": "acc_02"}),
    ("Lolita", {"hair": "hair_03", "dress": "dress_01",
               "shoes": "shoes_01", "acc": "acc_03"}),
    ("出游", {"hair": "hair_06", "top": "top_06", "bottom": "bottom_01",
             "shoes": "shoes_03", "acc": "acc_04"}),
    ("正装", {"hair": "hair_05", "dress": "dress_05",
             "shoes": "shoes_03", "acc": "acc_02"}),
]


def render(pick):
    cv = Image.new("RGBA", tuple(M["canvas"]), (255, 255, 255, 255))
    cv.alpha_composite(Image.open(os.path.join(ROOT, "full/body/body_00.png")).convert("RGBA"))
    for slot in ORDER:
        if slot in ("top", "bottom") and pick.get("dress"):
            continue  # 穿连衣裙时 hide top+bottom
        n = pick.get(slot)
        if not n:
            continue
        path = os.path.join(ROOT, "full", M["items"][n]["cat"], n + ".png")
        cv.alpha_composite(Image.open(path).convert("RGBA"))
    return cv


def main():
    w, h = 380, 570
    pad, bar = 10, 30
    sheet = Image.new("RGB", (len(LOOKS) * (w + pad) + pad, h + bar + 2 * pad), (250, 250, 252))
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
    except Exception:
        font = ImageFont.load_default()
    for i, (name, pick) in enumerate(LOOKS):
        im = render(pick)
        x = pad + i * (w + pad)
        sheet.paste(im.convert("RGB").resize((w, h), Image.LANCZOS), (x, pad))
        dr.text((x + 4, pad + h + 4), f"{name}  " + " / ".join(pick.values()),
                fill=(30, 30, 30), font=font)
    out = "/Users/zhujinhui/claudeProject5/dressup-game/搭配验证.png"
    sheet.save(out)
    print("搭配验证图:", out)


if __name__ == "__main__":
    main()
