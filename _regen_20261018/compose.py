#!/usr/bin/env python3
"""按 index.html 的 z 序合成试穿图，用于验证新素材。"""
import os
import sys

from PIL import Image, ImageDraw

ROOT = "/Users/zhujinhui/claudeProject5/dressup-game"
RES = os.path.join(ROOT, "res")
BODY = "res/body/body_00.png"

# 与 index.html 的 CATS 一致
Z = {"hair": 11, "shoes": 12, "top": 13, "bottom": 14, "dress": 15, "acc": 20}


def compose(sel, bg=(255, 255, 255)):
    canvas = Image.new("RGBA", (1024, 1536), bg + (255,))
    canvas.alpha_composite(Image.open(os.path.join(ROOT, BODY)).convert("RGBA"))
    dress = bool(sel.get("dress"))
    order = sorted(Z.items(), key=lambda kv: kv[1])
    for cat, _ in order:
        if dress and cat in ("top", "bottom"):
            continue
        v = sel.get(cat)
        if not v:
            continue
        p = os.path.join(RES, cat, f"{cat}_{v}.png")
        if os.path.exists(p):
            canvas.alpha_composite(Image.open(p).convert("RGBA"))
    return canvas.convert("RGB")


def sheet(rows, out, cw=300, ch=450, cols=4):
    r = (len(rows) + cols - 1) // cols
    sh = Image.new("RGB", (cw * cols, (ch + 22) * r), (244, 244, 244))
    d = ImageDraw.Draw(sh)
    for i, (label, sel) in enumerate(rows):
        t = compose(sel).resize((cw - 8, ch - 8), Image.LANCZOS)
        x, y = (i % cols) * cw + 4, (i // cols) * (ch + 22) + 20
        sh.paste(t, (x, y))
        d.text((x + 2, y - 15), label, fill=(200, 40, 40))
    sh.save(out)
    print("saved", out, sh.size)


if __name__ == "__main__":
    cases = [
        ("hair_02 + T恤/短裙", {"hair": "02", "top": "02", "bottom": "02", "shoes": "01"}),
        ("hair_06 + 毛衣/长裤", {"hair": "06", "top": "01", "bottom": "03", "shoes": "02"}),
        ("top_05 夹克 + 短裤", {"hair": "04", "top": "05", "bottom": "01", "shoes": "02"}),
        ("top_05 夹克 + 长裙", {"hair": "03", "top": "05", "bottom": "02", "shoes": "01"}),
        ("shoes_03 靴 + 短裙", {"hair": "01", "top": "03", "bottom": "02", "shoes": "03"}),
        ("shoes_03 靴 + 短裤", {"hair": "04", "top": "04", "bottom": "01", "shoes": "03"}),
        ("bottom_06 背带裤 + 白T", {"hair": "01", "top": "02", "bottom": "06", "shoes": "02"}),
        ("bottom_06 + 毛衣", {"hair": "05", "top": "01", "bottom": "06", "shoes": "01"}),
        ("dress_05 旗袍", {"hair": "02", "dress": "05", "shoes": "01"}),
        ("dress_05 + 草帽", {"hair": "03", "dress": "05", "shoes": "01", "acc": "01"}),
        ("acc_04 挎包", {"hair": "01", "top": "02", "bottom": "01", "shoes": "02", "acc": "04"}),
        ("acc_04 + 夹克", {"hair": "04", "top": "05", "bottom": "03", "shoes": "04", "acc": "04"}),
    ]
    sheet(cases, os.path.join(ROOT, "_regen_verify.png"))
