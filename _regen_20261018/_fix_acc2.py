"""acc_02 眼镜：镜片里残留了眼睛/皮肤（生成图镜片是透明的，差分把眼球当成了饰品），
看起来镜片上有两块棕色脏斑。做法是把两个镜片圆内部清空，只留镜框白的高光。
"""
import sys
sys.path.insert(0, 'tools')
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

OUT = '_regen_20261018'


def clear_lenses(img):
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(int)
    al = a[:, :, 3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gold = (r > 160) & (g > 120) & (b < 130) & (r - b > 55) & (al > 120)
    gold = ndimage.binary_closing(gold, structure=np.ones((5, 5)))
    # 镜框是环，填洞后得到两个实心圆盘
    disc = ndimage.binary_fill_holes(gold)
    # 镜片 = 圆盘内部（腐蚀掉环本身）
    lens = ndimage.binary_erosion(disc, iterations=4) & ~gold
    a[:, :, 3] = np.where(lens, 0, al)
    return Image.fromarray(a), int((lens & (al > 0)).sum())


def drop_specks(img, min_size=250):
    a = np.array(img.convert('RGBA'))
    al = a[:, :, 3]
    m = al > 120
    labels, n = ndimage.label(m)
    if n <= 1:
        return img, 0
    sizes = ndimage.sum(m, labels, range(1, n + 1))
    keep = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s >= min_size])
    a[:, :, 3] = np.where(keep, al, 0)
    return Image.fromarray(a), int((m & ~keep).sum())


def main():
    img = Image.open('res/acc/acc_02.png')
    img, n1 = clear_lenses(img)
    print(f'acc_02 清镜片 {n1} px')
    img, n2 = drop_specks(img)
    print(f'acc_02 删碎点 {n2} px')
    img.save(f'{OUT}/acc_02_build.png')

    sys.path.insert(0, '_regen_20261018')
    from compose import compose
    c = compose({'hair': '01', 'dress': '03', 'shoes': '01'}).convert('RGBA')
    c.alpha_composite(img.convert('RGBA'))
    c.convert('RGB').crop((280, 240, 760, 640)).resize((720, 600), Image.LANCZOS).save('/tmp/_acc02_fix.png')
    print('saved')


main()
