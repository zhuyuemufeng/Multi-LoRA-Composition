from PIL import Image, ImageDraw, ImageFont


def merger_image(image_paths, main_title, titles, descriptions, out_path):
    images = [Image.open(path) for path in image_paths]
    # 使用系统字体并指定字体大小
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    desc_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    main_title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)

    # 计算每张图片的宽高
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights) + 120  # 额外的高度用于标题和说明

    # 创建新图像（白色背景）
    new_image = Image.new('RGB', (total_width, max_height + 60), (255, 255, 255))
    draw = ImageDraw.Draw(new_image)

    # 绘制总标题
    main_title_width = draw.textlength(main_title, font=main_title_font)
    main_title_x = (total_width - main_title_width) / 2
    draw.text((main_title_x, 10), main_title, font=main_title_font, fill="black")

    # 初始化起始位置
    x_offset = 0
    y_offset = 60

    for i, img in enumerate(images):
        # 粘贴图片
        new_image.paste(img, (x_offset, y_offset + 40))

        # 计算标题和说明的宽度
        title_width = draw.textlength(titles[i], font=title_font)
        desc_width = draw.textlength(descriptions[i], font=desc_font)

        title_x = x_offset + (img.width - title_width) / 2
        desc_x = x_offset + (img.width - desc_width) / 2

        # 绘制每个图片的标题和说明
        draw.text((title_x, y_offset), titles[i], font=title_font, fill="black")
        draw.text((desc_x, y_offset + img.height + 50), descriptions[i], font=desc_font, fill="black")

        x_offset += img.width

    # 保存新图像
    new_image.save(out_path)
