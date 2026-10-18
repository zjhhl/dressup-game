#!/usr/bin/env python3
"""素材批处理：去水印 → 绿幕抠透明 → 归档到 res/"""
import glob
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """素体肤色像素 mask（用于跨图对齐）"""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (r > 180) & (g > 150) & (b > 120) & (r > g) & (g > b) & (r - b > 30)


def align_to_body(rgb: np.ndarray, body_skin: np.ndarray) -> np.ndarray:
    """与素体肤色区域互相关，返回校正平移后的 RGB（生成图体型/位置逐张漂移）"""
    m = skin_mask(rgb).astype(float)
    corr = fftconvolve(body_skin.astype(float), m[::-1, ::-1], mode="full")
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak[0] - m.shape[0] + 1
    dx = peak[1] - m.shape[1] + 1
    if dy == 0 and dx == 0:
        return rgb
    out = np.zeros_like(rgb)
    h, w = rgb.shape[:2]
    ys_src, ys_dst = max(0, -dy), max(0, dy)
    xs_src, xs_dst = max(0, -dx), max(0, dx)
    n_h = min(h - ys_src, h - ys_dst)
    n_w = min(w - xs_src, w - xs_dst)
    out[ys_dst:ys_dst + n_h, xs_dst:xs_dst + n_w] = rgb[ys_src:ys_src + n_h, xs_src:xs_src + n_w]
    return out

SRC = "/Users/zhujinhui/claudeProject5/dressup-game/assets"
DST = "/Users/zhujinhui/claudeProject5/dressup-game/res"


def remove_watermark(img: Image.Image) -> Image.Image:
    """右下角水印区域填绿（用图像顶部边缘的中位绿色）"""
    a = np.array(img)
    h, w = a.shape[:2]
    # 采样顶部 20 行的中位绿作为填充色
    fill = np.median(a[:20, :, :3].reshape(-1, 3), axis=0)
    # 水印在右下角；角色居中，此区域不碰角色
    y0, x0 = int(h * 0.915), int(w * 0.72)
    a[y0:, x0:, :3] = fill.astype(np.uint8)
    return Image.fromarray(a)


def inpaint_diffuse(rgb: np.ndarray, hole: np.ndarray, seed: np.ndarray,
                    unknown: np.ndarray = None,
                    iters: int = 45, sigma: float = 5.0) -> np.ndarray:
    """把洞里的颜色由四周向内扩散填满（反复模糊 + 还原已知像素）。

    直接平涂主色在大洞上会是一块死板色块；扩散能把周围发丝的明暗带进来。
    只在洞的包围盒里算，避免整图 45 次高斯模糊把内存打爆。

    两个坑：
    1. 洞里的原始像素是绿幕的绿（透明像素仍保留绿色 RGB），必须先刷成主色再扩散；
    2. 透明背景同样残留绿色 RGB，如果把它当「已知像素」参与扩散，绿色会顺着
       模糊渗进洞里，在假发外缘留一圈绿毛边。所以 unknown 要包含全部透明区，
       但只有 hole 会被置为不透明。"""
    ys, xs = np.nonzero(hole)
    if len(ys) == 0:
        return rgb
    if unknown is None:
        unknown = hole
    pad = 24
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    sub_unk = unknown[y0:y1, x0:x1]
    src = rgb[y0:y1, x0:x1].astype(np.float32)
    out = src.copy()
    out[sub_unk] = seed
    for _ in range(iters):
        blur = ndimage.gaussian_filter(out, sigma=(sigma, sigma, 0))
        out[sub_unk] = blur[sub_unk]
        out[~sub_unk] = src[~sub_unk]
    res = rgb.astype(np.float32)
    res[y0:y1, x0:x1] = out
    return np.clip(res, 0, 255).astype(int)


def cover_body_hair(img: Image.Image, hair_mask: np.ndarray,
                    body_skin_col: np.ndarray, body_hair_col: np.ndarray) -> Image.Image:
    """头发图层的三道清理（只作用在素体原发区 hair_mask 内，绝不碰脸）：

    1. **边缘去绿**：绿幕抗锯齿会在发丝边缘留一圈墨绿点线，把绿分量压回去。
    2. **补洞**：假发没盖满素体波波头的地方，底下的栗棕发会露出来（两层头发）。
       这块洞必须填，否则短一点/窄一点的假发两侧就会挂一圈棕发。
    3. **剔肤色带 / 剔旧发色残影**：假发外缘露出的头皮、模型保留的原发丝，
       分别按亮度和颜色剔掉。"""
    a = np.array(img.convert("RGBA"))
    rgb, alpha = a[:, :, :3].astype(int), a[:, :, 3].copy()
    r_, g_, b_ = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    greenish = (g_ - np.maximum(r_, b_)) > 14

    # 1) 边缘去绿
    spill = (alpha > 20) & (greenish & ((g_ - np.maximum(r_, b_)) <= 60))
    rgb[:, :, 1] = np.where(spill, np.maximum(r_, b_), g_)

    # 假发主色：发区内不透明、非绿像素的众数
    sel = hair_mask & (alpha > 20) & (~greenish)
    if not sel.any():
        return img
    q = rgb[sel] // 12 * 12
    keys, counts = np.unique(q, axis=0, return_counts=True)
    base = (keys[counts.argmax()] + 6)
    dark_wig = base.mean() < 120

    lum = rgb.mean(axis=2)

    # 注意：不能用「接近肤色」当判据 —— 发丛高光本身就是浅肉色，
    # 一判就把整片高光当成肤色带（头发 01/03/05 会被误伤到两万像素）。
    # 透明像素的 RGB 往往还是绿色（绿幕没清干净的边缘），
    # 所以「透明」这条不能跟 ~greenish 混在一起判，否则缺口永远补不上
    hole = (alpha <= 20) & hair_mask

    # 假发色和素体发色太接近时（比如栗棕假发对栗棕原发），
    # 「剔旧发色」无从区分，一判就把整片刘海高光当残影吃掉 —— 这种情况只补洞不动色
    similar = np.abs(base - body_hair_col).max() <= 60
    if similar:
        bad = np.zeros_like(hole)
    elif dark_wig:
        # 深色假发：亮度差足够大，用亮度剔掉底模露出的头皮
        bad = (lum > 128) & (~greenish) & hair_mask
    else:
        # 浅色假发：亮度分不开，只剔「旧发色残影」，容差收到 22，
        # 并要求离轮廓 3px 以上，否则会把假发自带的深棕描边一起剔掉
        inner = ndimage.binary_erosion(hair_mask & (alpha > 20), iterations=3)
        bad = (np.abs(rgb - body_hair_col).max(axis=2) <= 22) & inner & hair_mask

    n_fix = int(hole.sum() + bad.sum())
    if n_fix == 0:
        a[:, :, :3] = rgb.astype(np.uint8)
        return Image.fromarray(a)

    # 细碎的肤色带/残影：用最近的「好发色」像素填，保住明暗过渡
    if bad.any():
        valid = hair_mask & (alpha > 20) & (~greenish) & (~bad)
        if valid.any():
            ind = ndimage.distance_transform_edt(~valid, return_indices=True)[1]
            rgb[bad] = rgb[ind[0], ind[1], :][bad]
        else:
            rgb[bad] = base
        alpha[bad] = 255

    # 大块空洞：扩散修补（洞越大越需要，平涂会留死板色块）
    if hole.any():
        rgb = inpaint_diffuse(rgb, hole, base, unknown=(alpha <= 20))
        alpha[hole] = 255

    touch = hole | bad
    seam = ndimage.binary_dilation(touch, iterations=3) & ~touch & hair_mask
    if seam.any():
        blur = ndimage.gaussian_filter(rgb.astype(float), sigma=(1.5, 1.5, 0))
        rgb[seam] = blur[seam]

    a[:, :, :3] = rgb.astype(np.uint8)
    a[:, :, 3] = alpha
    print(f"     头发清理 补洞 {int(hole.sum())} px / 修边 {int(bad.sum())} px  深色={dark_wig}")
    return Image.fromarray(a)


SHOE_LINE = 0.83          # 鞋子图层的上沿，top/bottom/dress 都不能越过这条线


def crop_upper_body(img: Image.Image) -> Image.Image:
    """bottom 部件：裁掉腰线以上（自带的白T会盖住 top）"""
    a = np.array(img)
    h = a.shape[0]
    a[: int(h * 0.625), :, 3] = 0
    return Image.fromarray(a)


def crop_to_feet(img: Image.Image) -> Image.Image:
    """shoes 部件：生成图是全身像，裁掉脚踝以上（会盖住衣服图层）"""
    a = np.array(img)
    h = a.shape[0]
    a[: int(h * SHOE_LINE), :, 3] = 0
    return Image.fromarray(a)


def crop_above_shoes(img: Image.Image) -> Image.Image:
    """top/dress 部件：生成图自带整条腿和裸脚，而 top(13)/dress(15) 的 z 高于
    shoes(12)，那对裸脚会整只盖在鞋上，只剩一圈鞋底露在外面 —— 看起来像
    「鞋子是空心的」。衣服本来就到不了脚踝，这里直接按鞋子图层的上沿切平。"""
    a = np.array(img.convert("RGBA"))
    a[int(a.shape[0] * SHOE_LINE):, :, 3] = 0
    return Image.fromarray(a)


def trim_legs(img: Image.Image, body_rgb: np.ndarray) -> Image.Image:
    """bottom 部件：鞋子线以下整片切掉（含裤管/裸腿）。

    bottom(14) 也高于 shoes(12)。脚踝以下留着的话，短裙的裸腿会盖住鞋口
    （看起来鞋子是空心的）；长裤则是把整只脚包进去，鞋子永远穿不上。
    试着按与素体的差分区分「裸腿」和「裤管」，但 align_to_body 有整数像素
    残差，裸腿的 diff 落在 20~40，和布料幅值重叠，分不干净 —— 所以统一切。"""
    a = np.array(img.convert("RGBA"))
    a[int(a.shape[0] * SHOE_LINE):, :, 3] = 0
    return Image.fromarray(a)


def crop_bib_pants(img: Image.Image, body_rgb: np.ndarray) -> Image.Image:
    """bottom_06 背带裤专用：胸片与肩带在腰线以上，矩形裁切会把它们整块切掉。
    改为与素体差分——白打底衫、光腿都和素体一致会被删掉，只留背带裤本体。
    （bottom 的 z 比 top 高，胸片盖在上衣外面才是背带裤的正确穿法）"""
    a = np.array(img.convert("RGBA"))
    rgb, alpha = a[:, :, :3].astype(int), a[:, :, 3]
    diff = np.abs(rgb - body_rgb).max(axis=2)
    keep = (diff > 22) & (alpha > 0)
    keep = ndimage.binary_opening(keep, structure=np.ones((3, 3)), iterations=2)
    labels, n = ndimage.label(keep)
    if n:
        sizes = ndimage.sum(keep, labels, range(1, n + 1))
        keep = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s >= 400])
    keep[int(a.shape[0] * 0.895):] = False          # 丢掉脚踝以下
    a[:, :, 3] = np.where(keep, alpha, 0)
    return Image.fromarray(a)


def clear_head(img: Image.Image, head_mask: np.ndarray) -> Image.Image:
    """top/dress 部件：整块让开素体头部。

    生成图自带完整的素体（脸 + 栗棕波波头 + 白T），而 top/dress 的 z 序（13/15）
    高于 hair（11）。之前只按发色删头部像素，残下的高光发丝、深色描边和绿幕
    抗锯齿边会整片盖在假发上 —— 看上去就像「假发底下露出棕色短发」，
    其实是上衣图层画上去的。这里改成按素体头部轮廓整块清零，一个都不留。

    被清掉的区域底下是同位置的素体（脸和白T逐像素一致），所以不会露出空洞。"""
    a = np.array(img.convert("RGBA"))
    kill = ndimage.binary_dilation(head_mask, iterations=4)
    a[:, :, 3] = np.where(kill, 0, a[:, :, 3])
    return Image.fromarray(a)


def diff_against_body(img: Image.Image, body_rgb: np.ndarray, thresh: int = 28,
                      opening_iters: int = 3, min_size: int = 500) -> Image.Image:
    """acc 部件：生成图是全身素体+饰品。与素体差分，删掉与素体相同的像素，
    开运算去掉轮廓错位产生的细线残影，只留饰品本体"""
    a = np.array(img.convert("RGBA"))
    rgb, alpha = a[:, :, :3].astype(int), a[:, :, 3]
    diff = np.abs(rgb - body_rgb).max(axis=2)
    keep = (diff > thresh) & (alpha > 0)
    # 脸周按「接近素体发色」删除的做法已经去掉：
    # 素体头发与饰品同色时（帽檐的草黄、镜框的金、发箍的白）会被成片误删，
    # 草帽帽顶凭空缺一块、镜框断成几截，全是这条判据干的。
    # 素体本身 diff≈0 已被上面删掉，不需要再靠颜色删一遍。
    # 闭运算：把抗锯齿造成的 1~2px 细断口接上（饰品里的金线、白色箍带都很细）
    keep = ndimage.binary_closing(keep, structure=np.ones((3, 3)), iterations=1)
    if opening_iters:
        keep = ndimage.binary_opening(keep, structure=np.ones((3, 3)), iterations=opening_iters)
    # 丢弃小碎块
    labels, n = ndimage.label(keep)
    sizes = ndimage.sum(keep, labels, range(1, n + 1))
    keep = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s >= min_size])
    a[:, :, 3] = np.where(keep, alpha, 0)
    return Image.fromarray(a)


def chroma_key(img: Image.Image) -> Image.Image:
    """绿幕 → 透明，附去绿边。

    判据用「盈余绿」eg = g - max(r, b)，而不是 g>r+15 且 g>b+15：
    后者会把青绿/青色布料一起判成背景整片抠掉（旗袍 dress_05 就是这么被抠没的）。
    实测：绿幕 eg≈121，青绿布料 eg≈26，阈值放 55 可以干净分开。
    """
    a = np.array(img.convert("RGB")).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    eg = g - np.maximum(r, b)                       # 盈余绿
    green = (eg > 55) & (g > 60)
    alpha = np.where(green, 0, 255).astype(np.uint8)
    # 去绿边：只压盈余绿明显的像素，避免把青绿布料本身染成灰色
    spill = (~green) & (eg > 32)
    g_fixed = np.where(spill, np.minimum(g, np.maximum(r, b)), g)
    out = np.stack([r, g_fixed, b, alpha], axis=2).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def body_head_region(body_rgb: np.ndarray) -> np.ndarray:
    """素体整个头部轮廓（发 + 脸，填洞后连成一片）。top/dress 要整块让开这块。"""
    hair_col = np.median(body_rgb[160:340, 380:640].reshape(-1, 3), axis=0)
    m = np.abs(body_rgb - hair_col).max(axis=2) <= 62
    m[720:] = False
    m = ndimage.binary_closing(m, structure=np.ones((9, 9)))

    face = skin_mask(body_rgb)
    labels, n = ndimage.label(face)
    if n:
        sizes = ndimage.sum(face, labels, range(1, n + 1))
        face = ndimage.binary_fill_holes(np.isin(labels, [int(np.argmax(sizes)) + 1]))

    head = ndimage.binary_fill_holes(m | face)
    head = ndimage.binary_closing(head, structure=np.ones((13, 13)))
    head[720:] = False
    return head


def body_hair_region(body_rgb: np.ndarray) -> np.ndarray:
    """素体原发区掩膜。

    关键点：发丛的高光和深色描边跟基准发色差得很远，只按颜色取会留一堆洞
    （洞里的像素永远被当成「不属于发区」，于是假发边缘的肤色带永远补不上）。
    所以做法是「发色 ∪ 脸 → 填洞 → 再减掉脸」，得到实心的发环。"""
    head = body_head_region(body_rgb)
    face = skin_mask(body_rgb)
    labels, n = ndimage.label(face)
    if n:
        sizes = ndimage.sum(face, labels, range(1, n + 1))
        face = ndimage.binary_fill_holes(np.isin(labels, [int(np.argmax(sizes)) + 1]))

    hair = head & ~ndimage.binary_dilation(face, iterations=2)
    hair[720:] = False

    # 下巴以下：只认「真发色」，不能用填洞结果。
    # 脸的最大连通块到下巴就结束了，往下 head 会把脖子和肩颈一起吞进来，
    # 于是假发没盖到的地方（脖子正前方）被补洞填成假发色 —— 就是脖子上那块深色补丁。
    # 两侧垂下来的长发本身是栗棕色，m 为真，能保住。
    hair_col = np.median(body_rgb[160:340, 380:640].reshape(-1, 3), axis=0)
    m_hair = np.abs(body_rgb - hair_col).max(axis=2) <= 62
    m_hair = ndimage.binary_closing(m_hair, structure=np.ones((5, 5)))
    fys = np.nonzero(face)[0]
    chin = int(fys.max()) if len(fys) else int(hair.shape[0] * 0.41)
    hair[chin:] &= m_hair[chin:]

    labels, n = ndimage.label(hair)
    if n:
        sizes = ndimage.sum(hair, labels, range(1, n + 1))
        hair = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s > 3000])
    return hair


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*", "*.png")))
    print(f"共 {len(files)} 张")
    body_rgb = None
    body_skin = None
    body_hair = None
    body_head = None
    body_hair_col = np.array([134, 88, 74])
    body_skin_col = np.array([253, 221, 205])
    body_path = os.path.join(SRC, "body", "body_00.png")
    if os.path.exists(body_path):
        body_rgb = np.array(Image.open(body_path).convert("RGB")).astype(int)
        body_skin = skin_mask(body_rgb).astype(float)
        body_hair = body_hair_region(body_rgb)
        body_head = body_head_region(body_rgb)
        body_hair_col = np.median(body_rgb[160:340, 380:640].reshape(-1, 3), axis=0).astype(int)
        # 肤色取脸部中央（避开头发与描边）
        face_px = body_rgb[300:560, 360:670]
        m = skin_mask(face_px)
        if m.any():
            body_skin_col = np.median(face_px[m], axis=0).astype(int)
    for p in files:
        cat = os.path.basename(os.path.dirname(p))          # body/hair/top/...
        name = os.path.splitext(os.path.basename(p))[0]     # body_00 / hair_01 ...
        out_dir = os.path.join(DST, cat)
        os.makedirs(out_dir, exist_ok=True)
        img = Image.open(p)
        # 已定稿的成品（如 acc_01 草帽、acc_04 挎包）跳过再次处理：
        # 它们是在对齐后的坐标上单独抠好的透明图，再走一遍对齐会整体错位
        final_path = os.path.join(os.path.dirname(p), f"{name}_final.png")
        if os.path.exists(final_path):
            img = Image.open(final_path).convert("RGBA")
            out_path = os.path.join(out_dir, f"{name}.png")
            img.save(out_path, optimize=True)
            print(f"  {os.path.relpath(final_path, SRC)} -> {os.path.relpath(out_path, DST)}  (成品直出)")
            continue
        if cat != "body" and body_skin is not None:
            rgb = align_to_body(np.array(img.convert("RGB")), body_skin)
            img = Image.fromarray(rgb)
        img = remove_watermark(img)
        img = chroma_key(img)
        if cat == "hair" and body_hair is not None:
            img = cover_body_hair(img, body_hair, body_skin_col, body_hair_col)
        if cat == "bottom":
            if name == "bottom_06" and body_rgb is not None:
                img = crop_bib_pants(img, body_rgb)   # 背带裤：胸片在腰线以上
            else:
                img = crop_upper_body(img)
                img = trim_legs(img, body_rgb)
        if cat == "shoes":
            img = crop_to_feet(img)
        if cat in ("top", "dress"):
            if body_head is not None:
                img = clear_head(img, body_head)
            img = crop_above_shoes(img)
        if cat == "acc" and body_rgb is not None:
            # acc_04 挎包区域素体错位大，用更高阈值去残影；
            # acc_02 眼镜框细，开运算降为 1 次否则镜框被腐蚀掉
            if name == "acc_04":
                img = diff_against_body(img, body_rgb, thresh=50)
            elif name == "acc_02":
                img = diff_against_body(img, body_rgb, opening_iters=0, min_size=80)
            else:
                img = diff_against_body(img, body_rgb)
        out_path = os.path.join(out_dir, f"{name}.png")
        img.save(out_path, optimize=True)
        print(f"  {os.path.relpath(p, SRC)} -> {os.path.relpath(out_path, DST)}")


if __name__ == "__main__":
    main()
