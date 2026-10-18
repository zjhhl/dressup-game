#!/usr/bin/env python3
"""res/ 下 PNG 批量转 webp（透明保留，体积约降 80%），供网页加载"""
import glob
import os

from PIL import Image

RES = "/Users/zhujinhui/claudeProject5/dressup-game/res"


def main():
    files = sorted(glob.glob(os.path.join(RES, "*", "*.png")))
    total_png = total_webp = 0
    for p in files:
        out = os.path.splitext(p)[0] + ".webp"
        Image.open(p).convert("RGBA").save(out, "WEBP", quality=90, method=6)
        total_png += os.path.getsize(p)
        total_webp += os.path.getsize(out)
    print(f"{len(files)} 张: {total_png/1e6:.1f}MB -> {total_webp/1e6:.1f}MB")


if __name__ == "__main__":
    main()
