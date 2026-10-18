"""acc_03 兔耳发箍收尾：删掉发箍压出来的额头投影。

strip_brown 已经清掉大头发丝，但发箍在额头上留了一圈暖色投影 (237,189,165)，
颜色不够棕（r=237 超过棕色判据上限 215），漏在发箍下方，看起来像额头长了雀斑。
用「离发箍本体距离」删：白环 / 粉内耳 / 灰描边算本体，其余一概去掉。
"""
import sys
sys.path.insert(0, 'tools')
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

OUT = '_regen_20261018'


def keep_near_core(img, tol=4):
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(int)
    al = a[:, :, 3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    white = (lum > 222) & (sat < 30)
    pink = (r > 205) & (r - b > 22) & (np.abs(g - b) < 16)
    edge = (lum < 145) & (sat < 30)
    core = (white | pink | edge) & (al > 120)
    core = ndimage.binary_closing(core, structure=np.ones((5, 5)))
    dist = ndimage.distance_transform_edt(~core)
    kill = (dist > tol) & (al > 0)
    a[:, :, 3] = np.where(kill, 0, al)
    return Image.fromarray(a), int(kill.sum())


def strip_orange(img):
    """额头上还留着几道橙色发丝（r-b 80~100，比粉内耳的 35 高得多），单独按色相删。
    白环 (250,248,246) 和粉内耳 (240,205,205) 的 r-b 都 < 40，不受影响。"""
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(int)
    al = a[:, :, 3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    orange = (r - b > 55) & (g - b > 20) & (r > 195)
    kill = orange & (al > 0)
    a[:, :, 3] = np.where(kill, 0, al)
    return Image.fromarray(a), int(kill.sum())


def strip_forehead(img, tol=22):
    """删掉图层里带进来的整块额头。它和素体额头几乎同色（(247,223,203) vs
    (253,221,205)），叠上去看不出，但差分的边缘会在额头上留一圈淡淡的划痕。
    删掉后由素体自己提供额头，边缘就干净了。
    粉内耳 (240,205,205) 的 g-b=0，靠这个和肤色 (g-b≈18) 分开。"""
    a = np.array(img.convert('RGBA'))
    rgb = a[:, :, :3].astype(int)
    al = a[:, :, 3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    d = np.abs(rgb - np.array([253, 221, 205])).max(axis=2)
    skin = (d <= tol) & (g - b > 8) & (al > 0)
    a[:, :, 3] = np.where(skin, 0, al)
    return Image.fromarray(a), int(skin.sum())


def drop_specks(img, min_size=800):
    """删掉零碎的发丝/描边碎片。发箍主体、两只兔耳、两片粉内耳都是上千像素的整块，
    而那些从发箍边缘漏下来的发丝碎屑只有几十到几百像素。"""
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
    img = Image.open('res/acc/acc_03.png')
    img, n = keep_near_core(img)
    print(f'acc_03 删投影 {n} px')
    img, n2 = strip_orange(img)
    print(f'acc_03 删橙色发丝 {n2} px')
    img, n3 = strip_forehead(img)
    print(f'acc_03 删额头 {n3} px')
    img, n4 = drop_specks(img)
    print(f'acc_03 删碎片 {n4} px')
    img.save(f'{OUT}/acc_03_build.png')

    sys.path.insert(0, '_regen_20261018')
    from compose import compose
    CW = 380
    sh = Image.new('RGB', (CW * 3, 400), (240, 240, 240))
    d = ImageDraw.Draw(sh)
    for k, h in enumerate(['01', '02', '04']):
        c = compose({'hair': h, 'dress': '03', 'shoes': '01'}).convert('RGBA')
        c.alpha_composite(img.convert('RGBA'))
        im = c.convert('RGB').crop((280, 20, 760, 420))
        sh.paste(im.resize((CW, int(400 * CW / 480)), Image.LANCZOS), (k * CW, 0))
        d.text((k * CW + 6, 4), 'hair_' + h, fill=(200, 30, 30))
    sh.save('/tmp/_acc03_fix.png')
    print('saved')


main()
