#!/usr/bin/env python3
"""生成缩略图：把部件裁到对应部位再缩小，供分类面板展示
（整图缩略图看不清部件差别，发型/鞋这类必须裁到局部）"""
import glob
import os

from PIL import Image

RES = "/Users/zhujinhui/claudeProject5/dressup-game/res"
OUT = os.path.join(RES, "thumbs")

# 各类部件在 1024x1536 画布上的展示区域 (left, top, right, bottom)
CROP = {
    "hair":   (140, 10, 884, 700),
    "acc":    (140, 10, 884, 700),
    "top":    (190, 520, 834, 1180),
    "bottom": (200, 880, 824, 1420),
    "dress":  (170, 480, 854, 1470),
    "shoes":  (240, 1230, 784, 1536),
}
# 特例：挎包在腰侧，不在头部
CROP_OVERRIDE = {"acc_04": (230, 520, 820, 1180)}

THUMB_W = 300


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RES, "*", "*.webp")))
    n = 0
    for p in files:
        cat = os.path.basename(os.path.dirname(p))
        name = os.path.splitext(os.path.basename(p))[0]
        if cat not in CROP or name.endswith("_final"):
            continue
        box = CROP_OVERRIDE.get(name, CROP[cat])
        img = Image.open(p).convert("RGBA").crop(box)
        h = round(img.height * THUMB_W / img.width)
        img = img.resize((THUMB_W, h), Image.LANCZOS)
        img.save(os.path.join(OUT, f"{name}.webp"), "WEBP", quality=88, method=6)
        n += 1
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"{n} 张缩略图, 共 {size/1e6:.2f}MB")


if __name__ == "__main__":
    main()
