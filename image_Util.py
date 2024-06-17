from PIL import Image, ImageDraw, ImageFont
import os
import zipfile
from pathlib import Path


def merger_image(image_paths, main_title, titles, descriptions, out_path):
    images = [Image.open(path) for path in image_paths]
    # 使用系统字体并指定字体大小
    title_font = ImageFont.load_default(24)
    desc_font = ImageFont.load_default(18)
    main_title_font = ImageFont.load_default(36)

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


def zipDir(dirpath, outFullName):
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')

        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

def merge_images_vertically(image_dir, output_path):
    image_paths = get_all_files_in_folder(image_dir)
    images = [Image.open(img) for img in image_paths]

    # 获取所有图片的宽度和高度
    widths, heights = zip(*(img.size for img in images))

    # 计算合并后图片的宽度和高度
    max_width = max(widths)
    total_height = sum(heights)

    # 创建一张新的空白图片
    merged_image = Image.new('RGB', (max_width, total_height))

    # 将每张图片粘贴到新图片中
    y_offset = 0
    for img in images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.size[1]  # 增加偏移量，以便下一张图片粘贴到正确的位置

    # 保存合并后的图片
    merged_image.save(output_path)

    print(f"图片已成功竖向合并并保存到 {output_path}")


def get_all_files_in_folder(folder_path):
    all_files = []
    # 使用 Path 对象遍历目录及其子目录中的所有文件
    for file_path in Path(folder_path).rglob('*'):
        if file_path.is_file():
            all_files.append(str(file_path))
    return all_files