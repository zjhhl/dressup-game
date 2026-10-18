# -*- coding: utf-8 -*-
"""
素材后处理（对应 docs/素材生成描述.md 第九节 + 差异提取分层）

流程：
  A 抠图   绿幕 -> 透明 PNG（软边 alpha + 去绿边 spill + 清平台水印 + 去噪点）
  B 分层   生成图 - 素体 = 已定位的部件层（无需任何缩放/对齐运算）
  C 裁切   按品类取并集矩形，同品类用同一套裁切规则
  D 压缩   PNG optimize；超预算再降级量化
  E 归档   res/<品类>/<品类>_<编号>.png + res/manifest.json
"""
import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

SRC = "/Users/zhujinhui/claudeProject5/dressup-game/assets"
OUT = "/Users/zhujinhui/claudeProject5/dressup-game/res"
TMP = "/Users/zhujinhui/claudeProject5/dressup-game/.tmp_keyed"

# 平台水印固定位置（已实测 33 张 bbox 完全一致）
WM = (898, 1468, 1024, 1536)  # x0, y0, x1, y1

# 抠图阈值：与背景色的 L1 距离
HARD = 100   # <= 视为纯背景
SOFT = 175   # >= 视为纯前景

# 差异提取阈值
DIFF_TH = 12

CATS = {
    "body":   ["body_00"],
    "hair":   ["hair_01", "hair_02", "hair_03", "hair_04", "hair_05", "hair_06"],
    "top":    ["top_01", "top_02", "top_03", "top_04", "top_05", "top_06"],
    "bottom": ["bottom_01", "bottom_02", "bottom_03", "bottom_04", "bottom_05", "bottom_06"],
    "dress":  ["dress_01", "dress_02", "dress_03", "dress_04", "dress_05", "dress_06"],
    "shoes":  ["shoes_01", "shoes_02", "shoes_03", "shoes_04"],
    "acc":    ["acc_01", "acc_02", "acc_03", "acc_04"],
}

# 文档第十节：图层 z 序（小 -> 大）
Z = {"shoes": 5, "bottom": 15, "body": 10, "top": 20, "dress": 20, "hair": 10, "acc": 30}


def bg_color(a):
    h, w = a.shape[:2]
    cs = [a[3, 3], a[3, w - 4], a[h - 4, 3], a[h - 4, w - 4]]
    return np.array([float(np.mean([c[i] for c in cs])) for i in range(3)])


def key(im):
    """绿幕 -> RGBA。

    用「绿幕优势度」ge = g - (r+b)/2 而不是「与背景色的距离」来算 alpha：
    距离法会把深色前景（黑夹克与绿幕的 L1 距离只有 159）误判成半透明背景。
    返回 (rgba uint8, alpha float 0..1)
    """
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    bg = bg_color(a)
    ge_bg = bg[1] - (bg[0] + bg[2]) / 2.0            # 背景的绿优势度 ~131
    ge = a[:, :, 1] - (a[:, :, 0] + a[:, :, 2]) / 2.0
    alpha = np.clip(1.0 - ge / ge_bg, 0.0, 1.0)
    alpha[ge >= ge_bg * 0.98] = 0.0                   # 明确是背景

    # 绿色衣物修复（青绿色旗袍本身偏绿，会被当成半背景）：
    # 真正的绿色衣物是一整块，腐蚀 2 圈后仍有核心；抗锯齿边缘只有 1~2px，腐蚀后归零
    amb = (alpha >= 0.35) & (alpha <= 0.90)
    core = ndimage.binary_erosion(amb, np.ones((5, 5)))
    if core.any():
        lab, n = ndimage.label(amb)
        keep = np.unique(lab[core])
        keep = keep[keep > 0]
        alpha[np.isin(lab, keep)] = 1.0

    # 抹掉绿幕上的极轻微噪声（约 0.6% 的像素会有 0.02~0.4 的假 alpha，
    # 叠在别的图层下会显出淡淡的脏痕）。绿色衣物修复之后再做，不会误伤旗袍。
    alpha[alpha < 0.45] = 0.0

    # 清平台水印
    alpha[WM[1]:WM[3], WM[0]:WM[2]] = 0.0

    # 去绿边：只对仍处于半透明的像素做，把绿优势压到接近 0
    g, r, b = a[:, :, 1], a[:, :, 0], a[:, :, 2]
    mid = (r + b) / 2.0
    fix = alpha < 1.0
    g = np.where(fix, mid + 0.15 * (g - mid), g)
    a[:, :, 1] = g

    # 去噪点：丢弃面积 < 40px 的连通域
    m = alpha > 0.5
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        drop = np.isin(lab, np.where(np.array(sizes) < 40)[0] + 1)
        alpha[drop] = 0.0

    rgba = np.dstack([a.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return rgba, alpha


def extract(item_rgba, item_a, body_rgba, body_a):
    """生成图 - 素体 = 部件层"""
    d = np.abs(item_rgba[:, :, :3].astype(np.int16)
               - body_rgba[:, :, :3].astype(np.int16)).max(axis=2)
    both = (item_a > 0.85) & (body_a > 0.85)
    outside = (item_a > 0.90) & (body_a < 0.15)      # 素体轮廓之外的新内容（长发/帽檐）
    mask = (both & (d > DIFF_TH)) | outside
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))     # 去 1~2px 漂移细线
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))     # 合小缝
    lab, n = ndimage.label(mask)
    if n > 1:  # 去碎块（模型在无关部位的细微漂移会形成小碎块，800px 能挡住）
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        mask = np.isin(lab, np.where(np.array(sizes) >= 800)[0] + 1)

    out = item_rgba.copy()
    out[:, :, 3] = np.where(mask, item_rgba[:, :, 3], 0)
    return out, int(mask.sum())


def bbox(a, thr=16):
    m = a[:, :, 3] > thr
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(TMP, exist_ok=True)

    # ---------- A 抠图 ----------
    print("[A] 抠图：绿幕 -> 透明")
    keyed = {}
    for cat, names in CATS.items():
        for n in names:
            p = os.path.join(SRC, {"body": "body"}.get(cat, cat), n + ".png")
            rgba, alpha = key(Image.open(p))
            keyed[n] = (rgba, alpha)
            d = os.path.join(TMP, cat)
            os.makedirs(d, exist_ok=True)
            Image.fromarray(rgba, "RGBA").save(os.path.join(d, n + ".png"))
    print(f"    完成 {len(keyed)} 张，水印区已清除")

    body_rgba, body_a = keyed["body_00"]

    # ---------- B 差异提取 ----------
    print("[B] 差异提取：生成图 - 素体 = 已定位部件层")
    layers, px = {}, {}
    for cat, names in CATS.items():
        for n in names:
            if cat == "body":
                layers[n] = body_rgba
                px[n] = int((body_a > 0.85).sum())
                continue
            rgba, alpha = keyed[n]
            lay, cnt = extract(rgba, alpha, body_rgba, body_a)
            layers[n] = lay
            px[n] = cnt
    for cat, names in CATS.items():
        med = int(np.median([px[n] for n in names]))
        if cat == "hair":
            med = int(np.median([px[n] for n in names if n != "hair_01"]))  # hair_01 与素体同发色，不参与基线
        for n in names:
            flag = ""
            if cat != "body":
                if px[n] < med * 0.35:
                    flag = "  ⚠️ 像素远低于同类中位，可能提取不足"
                elif px[n] > med * 2.5:
                    flag = "  ⚠️ 像素远高于同类中位，可能混入了非本件内容"
            print(f"    {n:<10}{px[n]:>8} px{flag}")

    # ---------- C 裁切：同品类同一矩形 ----------
    print("[C] 裁切：按品类取并集矩形")
    crop = {}
    for cat, names in CATS.items():
        if cat == "body":
            crop[cat] = (0, 0, 1024, 1536)
            continue
        boxes = [bbox(layers[n]) for n in names]
        boxes = [b for b in boxes if b]
        x0 = min(b[0] for b in boxes) - 2
        y0 = min(b[1] for b in boxes) - 2
        x1 = max(b[2] for b in boxes) + 2
        y1 = max(b[3] for b in boxes) + 2
        crop[cat] = (max(0, x0), max(0, y0), min(1024, x1), min(1536, y1))
        print(f"    {cat:<8}{crop[cat]}  尺寸 {crop[cat][2]-crop[cat][0]}x{crop[cat][3]-crop[cat][1]}")

    # ---------- D+E 压缩与归档 ----------
    print("[D/E] 归档 res/")
    manifest = {"canvas": [1024, 1536], "z": Z,
                "render_order": ["body", "shoes", "hair", "bottom", "top", "dress", "acc"],
                "hide_when_dress": ["top", "bottom"],
                "note": "素体是 z=0 的底；鞋必须画在素体之上，否则会被素体的脚盖住"
                        "（文档第十节把 shoes 写成 5、素体身体写成 10，照数字顺序画鞋会消失）。",
                "items": {}}
    for cat, names in CATS.items():
        dc = os.path.join(OUT, cat)
        df = os.path.join(OUT, "full", cat)
        os.makedirs(dc, exist_ok=True)
        os.makedirs(df, exist_ok=True)
        for n in names:
            im = Image.fromarray(layers[n], "RGBA")
            im.save(os.path.join(df, n + ".png"), optimize=True)          # 整层（已定位）
            x0, y0, x1, y1 = crop[cat]
            im.crop((x0, y0, x1, y1)).save(os.path.join(dc, n + ".png"), optimize=True)
            manifest["items"][n] = {"cat": cat, "z": Z[cat], "px": px[n],
                                    "crop": [x0, y0, x1 - x0, y1 - y0],
                                    "full": f"full/{cat}/{n}.png",
                                    "cropped": f"{cat}/{n}.png"}
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    def total_size():
        return sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(OUT) for f in fs)

    BUDGET = 4 * 1024 * 1024
    total = total_size()
    print(f"    原始 PNG 总体积 {total/1024/1024:.2f} MB")
    if total > BUDGET:  # 超 4MB 主包上限 -> 降色到 256 色重存
        print("    超出 4 MB，执行 256 色量化压缩")
        for cat, names in CATS.items():
            for n in names:
                for sub in (f"{cat}/{n}.png", f"full/{cat}/{n}.png"):
                    p = os.path.join(OUT, sub)
                    im = Image.open(p).convert("RGBA")
                    if im.getbbox():
                        try:
                            im.quantize(colors=192, method=Image.FASTOCTREE).save(p, optimize=True)
                        except Exception:
                            im.save(p, optimize=True)
        total = total_size()
    print(f"    最终 res/ 总体积 {total/1024/1024:.2f} MB（主包上限 4 MB）")

    # ---------- 审计图：素体 + 每层各合成一格 ----------
    audit_path = "/Users/zhujinhui/claudeProject5/dressup-game/分层审计.png"
    names = [n for c in CATS for n in CATS[c]]
    cols, th, pad, bar = 7, 260, 8, 22
    tw = int(th * 1024 / 1536)
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + bar + pad) + pad), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 15)
    except Exception:
        font = ImageFont.load_default()
    body_im = Image.fromarray(body_rgba, "RGBA")
    for i, n in enumerate(names):
        canvas = Image.new("RGBA", (1024, 1536), (255, 255, 255, 255))
        canvas.alpha_composite(body_im)
        canvas.alpha_composite(Image.fromarray(layers[n], "RGBA"))
        cell = canvas.convert("RGB").resize((tw, th), Image.LANCZOS)
        x = pad + (i % cols) * (tw + pad)
        y = pad + (i // cols) * (th + bar + pad)
        sheet.paste(cell, (x, y))
        dr.text((x + 2, y + th + 2), n, fill=(20, 20, 20), font=font)
    sheet.save(audit_path)
    print("    审计图:", audit_path)


if __name__ == "__main__":
    main()
