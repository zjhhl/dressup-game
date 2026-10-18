# -*- coding: utf-8 -*-
"""出图后处理前的数据体检：尺寸、绿幕色值、角色对齐、绿幕占比。"""
import os
import numpy as np
from PIL import Image

ROOT = "/Users/zhujinhui/claudeProject5/dressup-game/assets"
CATS = ["body", "hair", "top", "bottom", "dress", "shoes", "acc"]


def load(p):
    return np.array(Image.open(p).convert("RGB")).astype(np.int16)


def bbox_of_bg(a, bg, tol):
    """返回非背景像素的包围盒（即角色/前景范围）"""
    d = np.abs(a - bg).sum(axis=2)
    m = d > tol
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main():
    files = []
    for c in CATS:
        d = os.path.join(ROOT, c)
        for f in sorted(os.listdir(d)):
            if f.endswith(".png"):
                files.append((c, f[:-4], os.path.join(d, f)))

    base = load(os.path.join(ROOT, "body", "body_00.png"))
    h, w = base.shape[:2]
    corners = [tuple(base[2, 2]), tuple(base[2, w - 3]), tuple(base[h - 3, 2]), tuple(base[h - 3, w - 3])]
    bg = tuple(int(round(sum(c[i] for c in corners) / 4)) for i in range(3))
    print("素体尺寸:", w, "x", h)
    print("四角采样:", corners)
    print("背景基准色:", bg)

    bb0 = bbox_of_bg(base, np.array(bg), 60)
    print("素体前景包围盒:", bb0)
    print()
    print(f"{'编号':<10}{'尺寸':<12}{'前景bbox':<28}{'与素体偏差':<16}{'绿幕占比'}")
    bad = []
    for c, name, p in files:
        a = load(p)
        if a.shape[:2] != base.shape[:2]:
            bad.append((name, "尺寸不一致 " + str(a.shape[:2])))
            continue
        bb = bbox_of_bg(a, np.array(bg), 60)
        dev = tuple(bb[i] - bb0[i] for i in range(4))
        d = np.abs(a - np.array(bg)).sum(axis=2)
        frac = float((d <= 60).mean())
        print(f"{name:<10}{str(a.shape[1])+'x'+str(a.shape[0]):<12}{str(bb):<28}{str(dev):<16}{frac:.3f}")
    print()
    print("异常:", bad if bad else "无")


if __name__ == "__main__":
    main()
