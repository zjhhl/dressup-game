"""草帽图层修复（第 3 版）

问题：帽檐内缘有抠图洞（头发从洞里透出，像「帽子外面还有头发」），帽檐下缘
还挂着一块额头（差分时被当饰品保留）。

判据要点：草黄和肤色的 RGB 只差在蓝通道，靠绝对色差分不开，但 b/g 比值差别明显——
草黄 b/g≈0.70，肤色 b/g≈0.88。所以用 b < g*0.82 把草檐所有明暗色阶（含两侧卷边）
一把捞全，再用「到帽子本体的距离」删掉挂在帽檐外的额头。
"""
import sys
sys.path.insert(0, 'tools')
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import process_assets as P

A = 'res/acc/acc_01.png'
SKIN = np.array([253, 221, 205])


def seal_holes(img):
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(np.float32)
    al = a[:, :, 3]
    solid = al > 200
    fill = ndimage.binary_fill_holes(ndimage.binary_closing(solid, structure=np.ones((7, 7))))
    hole = fill & (al < 20)
    if not hole.any():
        return img, 0
    seed = np.median(rgb[solid], axis=0)
    out = rgb.copy()
    out[hole] = seed
    src = rgb.copy()
    for _ in range(50):
        blur = ndimage.gaussian_filter(out, sigma=(4, 4, 0))
        out[hole] = blur[hole]
        # 透明像素的 RGB 还留着头发/肤色/绿幕，不能当已知色，否则会渗进来
        out[solid] = src[solid]
    a[:, :, :3] = np.clip(out, 0, 255).astype(np.uint8)
    a[:, :, 3] = np.where(hole, 255, al)
    return Image.fromarray(a), int(hole.sum())


def strip_skin(img, tol=42):
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(int)
    al = a[:, :, 3]
    d = np.abs(rgb - SKIN).max(axis=2)
    skin = (d <= tol) & (al > 0)
    a[:, :, 3] = np.where(skin, 0, al)
    return Image.fromarray(a), int(skin.sum())


def drop_outside_brim(img, tol=10):
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(int)
    al = a[:, :, 3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    straw = (r > 70) & (g > 55) & (r > g + 8) & (b < g * 0.82)
    pink = (r > 200) & (b > 140) & (r - g > 35) & (b > g)
    core = (straw | pink) & (al > 120)
    core = ndimage.binary_closing(core, structure=np.ones((5, 5)))

    dist = ndimage.distance_transform_edt(~core)
    kill = (dist > tol) & (al > 0) & (~core)
    a[:, :, 3] = np.where(kill, 0, al)
    return Image.fromarray(a), int(kill.sum())


def degreen(img):
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(int)
    al = a[:, :, 3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    eg = g - np.maximum(r, b)
    spill = (al > 0) & (eg > 4)
    rgb[:, :, 1] = np.where(spill, np.maximum(r, b), g)
    a[:, :, :3] = rgb.astype(np.uint8)
    return Image.fromarray(a)


def widen_crown(img, k=0.16, pivot=310, span=250):
    """帽冠横向加宽（越靠近帽顶加得越多，帽檐不动）。

    帽冠原本比头发窄一圈，戴上去头发会从帽冠两侧鼓出来，看起来像「帽子外面还有
    头发」。纵向渐变的横向拉伸能让帽冠包住头，又不改变帽檐的落点（帽檐压眉的位置
    是调好的）。"""
    src = np.array(img.convert('RGBA')).astype(np.float32)
    h, w = src.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = 1 + k * np.clip((pivot - yy) / span, 0, 1)
    xs = (w / 2) + (xx - w / 2) / s
    out = np.stack([ndimage.map_coordinates(src[:, :, c], [yy, xs], order=3, mode='nearest')
                    for c in range(4)], axis=2)
    out[:, :, 3] = np.where(out[:, :, 3] > 110, 255, 0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    img = Image.open(A)
    img, nh = seal_holes(img)
    print(f'填内部洞 {nh} px')
    img, ns = strip_skin(img)
    print(f'删肤色残留 {ns} px')
    img, nb = drop_outside_brim(img)
    print(f'删帽檐外残留 {nb} px')
    img = degreen(img)
    # 收掉删肉色时在帽檐下缘留下的尖锐小缺口——这些缺口会让头发从缝里冒出来
    a = np.array(img.convert('RGBA'))
    al = a[:, :, 3]
    m = ndimage.binary_closing(al > 120, structure=np.ones((7, 7)))
    m = ndimage.binary_fill_holes(m)
    a[:, :, 3] = np.where(m & (al > 0), 255, a[:, :, 3])
    img = Image.fromarray(a)
    img = widen_crown(img)
    # 删掉帽檐下缘的碎点：删肉色时描边被切成一串孤立小黑点，放大后像毛边
    a = np.array(img.convert('RGBA'))
    al = a[:, :, 3]
    m = al > 120
    labels, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, labels, range(1, n + 1))
        keep = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s >= 60])
        a[:, :, 3] = np.where(keep, al, 0)
    img = Image.fromarray(a)
    img.save('_regen_20261018/acc_01_build.png')

    acc = img.convert('RGBA')
    d = Image.new('RGBA', (1024, 1536), (255, 0, 255, 255))
    d.alpha_composite(acc)
    d.convert('RGB').crop((30, 20, 1000, 620)).resize((560, 347), Image.LANCZOS).save('/tmp/_h3_alone.png')

    sys.path.insert(0, '_regen_20261018')
    from compose import compose
    c = compose({'hair': '02', 'dress': '03', 'shoes': '01'}).convert('RGBA')
    c.alpha_composite(acc)
    c.convert('RGB').crop((30, 20, 1000, 620)).resize((560, 347), Image.LANCZOS).save('/tmp/_h3_worn.png')

    sh = Image.new('RGB', (1130, 360), (240, 240, 240))
    sh.paste(Image.open('/tmp/_h3_alone.png'), (0, 8))
    sh.paste(Image.open('/tmp/_h3_worn.png'), (566, 8))
    sh.save('/tmp/_hat_fix3.png')
    print('saved')


main()
