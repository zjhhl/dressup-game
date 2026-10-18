#!/usr/bin/env python3
"""对比新旧绿幕判据对 33 张素材 alpha 的影响，确认改动是安全的。"""
import glob
import os

import numpy as np
from PIL import Image

SRC = "/Users/zhujinhui/claudeProject5/dressup-game/assets"


def old_key(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return (g > r + 15) & (g > b + 15) & (g > 60)


def new_key(a):
    """盈余绿：g - max(r,b)。纯绿幕 #00FF00 = 255；青绿 (50,150,130) 只有 20。"""
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    eg = g - np.maximum(r, b)
    return (eg > 50) & (g > 60)


rows = []
for p in sorted(glob.glob(os.path.join(SRC, "*", "*.png"))):
    if "_backup" in p:
        continue
    a = np.array(Image.open(p).convert("RGB")).astype(np.int16)
    o = old_key(a)
    n = new_key(a)
    bg_old = o.mean() * 100
    bg_new = n.mean() * 100
    diff = np.logical_xor(o, n).mean() * 100
    rows.append((os.path.relpath(p, SRC), bg_old, bg_new, diff))

print("%-26s %8s %8s %8s" % ("asset", "旧判背景%", "新判背景%", "差异%"))
for name, bo, bn, d in rows:
    flag = "  <<< 变化大" if d > 0.5 else ""
    print("%-26s %8.2f %8.2f %8.3f%s" % (name, bo, bn, d, flag))

print()
print("差异 >0.5%% 的素材数:", sum(1 for r in rows if r[3] > 0.5))
