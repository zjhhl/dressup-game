# -*- coding: utf-8 -*-
"""把本批生成的素材按「品类_编号」归档，并输出总览核对图。"""
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

SRC = "/Users/zhujinhui/.workbuddy/plugins/data/mcp-miora/miora-media"
ROOT = "/Users/zhujinhui/claudeProject5/dressup-game/assets"

# (编号, 品类, 源文件名)
ITEMS = [
    ("body_00",   "body",   "miora_text_to_image-1790754591260-0-438d5e112e7b.png"),
    ("hair_01",   "hair",   "miora_edit_image-1790754634939-0-2b301c933fca.png"),
    ("hair_02",   "hair",   "miora_edit_image-1790754636980-0-71ac2f8932f0.png"),
    ("hair_03",   "hair",   "miora_edit_image-1790754634752-0-734a5920069e.png"),
    ("hair_04",   "hair",   "miora_edit_image-1790754633630-0-68fc222b304e.png"),
    ("hair_05",   "hair",   "miora_edit_image-1790754636465-0-0ba3f0ecd4b8.png"),
    ("hair_06",   "hair",   "miora_edit_image-1790754635959-0-607ea747e90e.png"),
    ("top_01",    "top",    "miora_edit_image-1790754690060-0-bbe9e8f10a6d.png"),
    ("top_02",    "top",    "miora_edit_image-1790754686854-0-3231e67aebab.png"),
    ("top_03",    "top",    "miora_edit_image-1790754687842-0-8f39e101001e.png"),
    ("top_04",    "top",    "miora_edit_image-1790754685542-0-904285f5b5f3.png"),
    ("top_05",    "top",    "miora_edit_image-1790754684500-0-d21ebfc69836.png"),
    ("top_06",    "top",    "miora_edit_image-1790754686795-0-b2a055b25dd0.png"),
    ("bottom_01", "bottom", "miora_edit_image-1790754683537-0-257a641081bd.png"),
    ("bottom_02", "bottom", "miora_edit_image-1790754683320-0-22bb7740289d.png"),
    ("bottom_03", "bottom", "miora_edit_image-1790754686821-0-fa5f4640387d.png"),
    ("bottom_04", "bottom", "miora_edit_image-1790754687913-0-34cb52a04437.png"),
    ("bottom_05", "bottom", "miora_edit_image-1790754684243-0-2d1d853b3f84.png"),
    ("bottom_06", "bottom", "miora_edit_image-1790754685659-0-7ede6a339199.png"),
    ("dress_01",  "dress",  "miora_edit_image-1790754751415-0-a227a5195e33.png"),
    ("dress_02",  "dress",  "miora_edit_image-1790754744405-0-6b18b4fc587f.png"),
    ("dress_03",  "dress",  "miora_edit_image-1790754744754-0-0a5c23e8aa2d.png"),
    ("dress_04",  "dress",  "miora_edit_image-1790754745452-0-1b8f34427c97.png"),
    ("dress_05",  "dress",  "miora_edit_image-1790754751532-0-73571f6c38be.png"),
    ("dress_06",  "dress",  "miora_edit_image-1790754742914-0-0ceac0644f17.png"),
    ("shoes_01",  "shoes",  "miora_edit_image-1790754743703-0-0a90de3c84ad.png"),
    ("shoes_02",  "shoes",  "miora_edit_image-1790754741615-0-3a69641402a9.png"),
    ("shoes_03",  "shoes",  "miora_edit_image-1790754750196-0-b552fb4618c5.png"),
    ("shoes_04",  "shoes",  "miora_edit_image-1790754749664-0-938661e2fcc9.png"),
    ("acc_01",    "acc",    "miora_edit_image-1790754747960-0-8a9944a8aaf5.png"),
    ("acc_02",    "acc",    "miora_edit_image-1790754747546-0-91fe68e0e8b7.png"),
    ("acc_03",    "acc",    "miora_edit_image-1790754746497-0-4b0aa1285016.png"),
    ("acc_04",    "acc",    "miora_edit_image-1790754749626-0-8753936c113a.png"),
]

LABEL = {
    "body_00": "body 素体",
    "hair_01": "hair_01 栗棕短发", "hair_02": "hair_02 黑长直",
    "hair_03": "hair_03 粉棕双马尾", "hair_04": "hair_04 亚麻波波头",
    "hair_05": "hair_05 紫低丸子", "hair_06": "hair_06 金大波浪",
    "top_01": "top_01 粉针织", "top_02": "top_02 草莓T",
    "top_03": "top_03 水手服", "top_04": "top_04 奶黄泡泡袖",
    "top_05": "top_05 皮夹克", "top_06": "top_06 紫卫衣",
    "bottom_01": "bottom_01 牛仔短裤", "bottom_02": "bottom_02 藏青百褶",
    "bottom_03": "bottom_03 米白长裤", "bottom_04": "bottom_04 红黑格裙",
    "bottom_05": "bottom_05 黑打底裤", "bottom_06": "bottom_06 卡其背带裤",
    "dress_01": "dress_01 白Lolita", "dress_02": "dress_02 红波点吊带",
    "dress_03": "dress_03 淡紫长裙", "dress_04": "dress_04 黄背带裙",
    "dress_05": "dress_05 青绿旗袍", "dress_06": "dress_06 粉白纱裙",
    "shoes_01": "shoes_01 红玛丽珍", "shoes_02": "shoes_02 白运动鞋",
    "shoes_03": "shoes_03 棕短靴", "shoes_04": "shoes_04 粉凉鞋",
    "acc_01": "acc_01 草编帽", "acc_02": "acc_02 金圆框眼镜",
    "acc_03": "acc_03 兔耳发箍", "acc_04": "acc_04 棕挎包",
}


def find_font(size):
    for p in ["/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Medium.ttc",
              "/Library/Fonts/Arial Unicode.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    os.makedirs(ROOT, exist_ok=True)
    meta = []
    for name, cat, fn in ITEMS:
        src = os.path.join(SRC, fn)
        d = os.path.join(ROOT, cat)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, name + ".png")
        shutil.copyfile(src, dst)
        meta.append((name, cat, dst))

    # 总览图：每格等比缩放到高 300，白底，下方标名
    cols, th, pad, bar = 7, 300, 10, 26
    font = find_font(18)
    tw = int(th * 1024 / 1536)
    rows = (len(meta) + cols - 1) // cols
    cw, ch = tw + pad, th + bar + pad
    sheet = Image.new("RGB", (cols * cw + pad, rows * ch + pad), (245, 246, 248))
    dr = ImageDraw.Draw(sheet)
    for i, (name, cat, path) in enumerate(meta):
        im = Image.open(path).convert("RGB").resize((tw, th), Image.LANCZOS)
        x = pad + (i % cols) * cw
        y = pad + (i // cols) * ch
        sheet.paste(im, (x, y))
        dr.text((x + 2, y + th + 3), LABEL.get(name, name), fill=(20, 20, 20), font=font)
    out = os.path.join(ROOT, "素材总览.png")
    sheet.save(out)
    print("归档", len(meta), "张 →", ROOT)
    print("总览图:", out, sheet.size)


if __name__ == "__main__":
    main()
