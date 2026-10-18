#!/usr/bin/env python3
"""头发图层残影清理：只把「位于发团内部、且颜色接近素体棕发」的像素
替换成最近的真发色像素。脸/眉/眼/外描边都在发团外，不会被误伤。"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = "/Users/zhujinhui/claudeProject5/dressup-game"


def dominant(pixels, k=512):
    """把像素量化后取众数，得到这一团的主色"""
    q = (pixels // 12 * 12)
    keys, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    return keys[counts.argmax()].astype(np.int16)


def clean_hair_ghost(wig_rgb, body_rgb, verbose=True):
    w = wig_rgb.astype(np.int16)
    b = body_rgb.astype(np.int16)
    h, width = w.shape[:2]

    body_hair = np.median(b[160:340, 380:640].reshape(-1, 3), axis=0).astype(np.int16)

    # 背景（绿幕）粗判，排除掉
    r, g, bl = w[:, :, 0], w[:, :, 1], w[:, :, 2]
    bg = (g - np.maximum(r, bl) > 55) & (g > 60)

    # 1) 发团：用「占画面主导的深色/亮色」不好定，改成用非背景区里
    #    与素体发色明显不同的像素求主色
    cand = (~bg) & (np.abs(w - body_hair).max(2) > 70)
    hair_col = dominant(w[cand])
    if verbose:
        print("   主发色 =", hair_col, " 素体发色 =", body_hair)

    # 2) 发团掩膜 + 闭运算 + 填洞（让贴在轮廓边上的棕色斑也算进内部）
    hair_mask = (np.abs(w - hair_col).max(2) <= 78) & (~bg)
    hair_mask = ndimage.binary_closing(hair_mask, structure=np.ones((11, 11)))
    hair_mask = ndimage.binary_fill_holes(hair_mask)
    hair_mask = ndimage.binary_erosion(hair_mask, iterations=2)

    # 3) 残影 = 发团内 且 颜色接近素体发色
    ghost = hair_mask & (np.abs(w - body_hair).max(2) <= 50)
    if verbose:
        print("   残影像素 =", int(ghost.sum()))

    if ghost.sum() == 0:
        return wig_rgb

    # 4) 用最近的非残影发色像素填
    valid = hair_mask & (~ghost)
    ind = ndimage.distance_transform_edt(~valid, return_indices=True)[1]
    filled = w[ind[0], ind[1], :].astype(np.int16)
    out = w.copy()
    out[ghost] = filled[ghost]
    return out.astype(np.uint8)


if __name__ == "__main__":
    body = np.array(Image.open(f"{ROOT}/assets/body/body_00.png").convert("RGB"))
    for tag in ("hair_02_v3", "hair_06_v3"):
        print("==", tag)
        w = np.array(Image.open(f"{ROOT}/_regen_20261018/{tag}.png").convert("RGB"))
        out = clean_hair_ghost(w, body)
        Image.fromarray(out).save(f"{ROOT}/_regen_20261018/{tag}_clean.png")
        # 对比图
        a = Image.fromarray(out).crop((120, 40, 900, 720)).resize((470, 410), Image.LANCZOS)
        bimg = Image.fromarray(w).crop((120, 40, 900, 720)).resize((470, 410), Image.LANCZOS)
        sh = Image.new("RGB", (952, 410), (245, 245, 245))
        sh.paste(bimg, (0, 0))
        sh.paste(a, (482, 0))
        sh.save(f"{ROOT}/_clean_{tag}.png")
        print("   saved", f"_clean_{tag}.png")
