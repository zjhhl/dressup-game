"""修 acc 三处问题：
1. acc_01 草帽：帽冠与帽檐交界的抠图洞（头发从洞里透出，看起来像「帽子外面还有头发」）
2. acc_03 兔耳发箍：图层里带进了大把头丝（棕色），z=20 会画在假发上面
3. acc_02 眼镜：y>900 的衣服/脚部残差
"""
import sys
sys.path.insert(0, 'tools')
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import process_assets as P

OUT = '_regen_20261018'


def seal_holes(img, max_hole=9000, seed=(0, 0, 0)):
    """填掉图层内部的透明小洞（用四周颜色扩散填），保留大开口。"""
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(np.float32)
    al = a[:, :, 3]
    solid = al > 200
    fill = ndimage.binary_fill_holes(ndimage.binary_closing(solid, structure=np.ones((7, 7))))
    cand = fill & (al < 20)
    labels, n = ndimage.label(cand)
    if n == 0:
        return img, 0
    sizes = ndimage.sum(cand, labels, range(1, n + 1))
    # fill_holes 只填「被实心区完全包围」的区域，因此 cand 里的洞全是帽子内部该补的；
    # 帽檐中心的露脸开口与外部连通，不会被算进来，无需额外过滤。
    hole = cand
    if not hole.any():
        return img, 0
    sub = hole
    out = rgb.copy()
    out[sub] = seed                      # 先把绿幕色刷掉，否则扩散会渗绿
    src = rgb.copy()
    for _ in range(40):
        blur = ndimage.gaussian_filter(out, sigma=(3, 3, 0))
        out[sub] = blur[sub]
        out[~sub] = src[~sub]
    a[:, :, :3] = np.clip(out, 0, 255).astype(np.uint8)
    a[:, :, 3] = np.where(sub, 255, al)
    return Image.fromarray(a), int(sub.sum())


def strip_brown(img, hair_col=np.array([134, 88, 74])):
    """删掉棕色发丝（只适用于白色/粉色头饰）。判据在 acc_03 上验证过：
    发丝(深棕/棕/米色高光)全部命中，发箍的白环、粉内耳、灰描边一个不碰。"""
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(int)
    al = a[:, :, 3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brown = (r > g + 6) & (g >= b - 4) & (r - b > 22) & (r < 215)
    kill = brown & (al > 0)
    a[:, :, 3] = np.where(kill, 0, al)
    return Image.fromarray(a), int(kill.sum())


def main():
    # ---- acc_01 草帽：填内部洞
    hat = Image.open('res/acc/acc_01.png')
    hat2, nh = seal_holes(hat, max_hole=2500)
    hat2.save(f'{OUT}/acc_01_build.png')
    print(f'acc_01 填洞 {nh} px')

    # ---- acc_03 兔耳发箍：删棕色发丝，再补一遍边缘
    a3 = Image.open('res/acc/acc_03.png')
    a3b, nk = strip_brown(a3)
    body = np.array(a3b.convert('RGBA'))
    al = body[:, :, 3] > 120
    al = ndimage.binary_closing(al, structure=np.ones((3, 3)))
    body[:, :, 3] = np.where(al, 255, 0)
    a3c = Image.fromarray(body)
    a3c.save(f'{OUT}/acc_03_build.png')
    print(f'acc_03 删发丝 {nk} px')

    # ---- acc_02 眼镜：去掉下半身残差
    a2 = np.array(Image.open('res/acc/acc_02.png').convert('RGBA'))
    cut = a2[:, :, 3]
    low = (cut > 0)
    low[:980] = False
    print(f'acc_02 下半身残留 {int(low.sum())} px')
    a2[:, :, 3] = np.where(low, 0, cut)
    Image.fromarray(a2).save(f'{OUT}/acc_02_build.png')

    # ---- 预览：三件按 z 序戴上
    sys.path.insert(0, '_regen_20261018'); from compose import compose
    import os
    cases = [('acc_01', '02', (240, 20, 800, 620)),
             ('acc_03', '02', (240, 0, 800, 560)),
             ('acc_02', '02', (300, 300, 740, 620))]
    CW = 420
    sh = Image.new('RGB', (CW * 3, 560), (242, 242, 242))
    d = ImageDraw.Draw(sh)
    for k, (acc, hair, box) in enumerate(cases):
        base = compose({'hair': hair, 'dress': '03', 'shoes': '01'})
        c = base.convert("RGBA").copy()
        c.alpha_composite(Image.open(f'{OUT}/{acc}_build.png').convert('RGBA'))
        im = c.convert('RGB').crop(box)
        w, h = im.size
        nh2 = min(int(h * CW / w), 520)
        sh.paste(im.resize((CW, nh2), Image.LANCZOS), (k * CW, 20))
        d.text((k * CW + 6, 4), acc + ' 修复后', fill=(200, 30, 30))
    sh.save('/tmp/_acc_fix1.png')
    print('saved /tmp/_acc_fix1.png')


main()
