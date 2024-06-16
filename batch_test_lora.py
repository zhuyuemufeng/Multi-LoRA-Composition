import argparse
from image_Util import merger_image
from adapter_lora import generate_image as adapter_generate_image
from fuse_lora_current import generate_image as fuse_current_generate_image
from fuse_lora_save import generate_image as fuse_save_generate_image
from base_lora import generate_image as base_generate_image
from utils import load_lora_info, get_prompt


def main(lora_type):
    image_style = ["anime", "reality"]
    lora_method = ["merge", "switch", "composite"]
    speed_type = ["LCM", "LCM", "TCD"]
    for style in image_style:
        lora_info = load_lora_info(style)
        init_prompt, negative_prompt = get_prompt(args.image_style)
        for lora_data in lora_info["character"]:
            lora_name = lora_data[id]
            lora_tagger = lora_data['trigger']
            prompt = init_prompt + ', ' + ', '.join(lora_tagger)
            print(f"prompt: {prompt}")
            images = []
            titles = []
            descriptions = []
            base_lora_image, base_lora_time = base_generate_image(f"/kaggle/input/lora-model/lora/{style}",
                                                                  lora_name, prompt, negative_prompt)
            base_image, base_time = base_generate_image("", "", prompt, negative_prompt)
            images.append(base_image)
            titles.append(f"base")
            descriptions.append(f"excuteTime: {base_time} s")
            images.append(base_lora_image)
            titles.append(f"base + lora")
            descriptions.append(f"excuteTime: {base_lora_time} s")
            for speed in speed_type:
                for method in lora_method:
                    if lora_type == "fuse_save":
                        generate_image, generate_time = fuse_save_generate_image(
                            f"/kaggle/input/lora-model/lora/{style}",
                            method, speed, lora_name, prompt, negative_prompt)
                    elif lora_type == "fuse":
                        generate_image, generate_time = fuse_current_generate_image(
                            f"/kaggle/input/lora-model/lora/{style}",
                            method, speed, lora_name, prompt, negative_prompt)
                    else:
                        generate_image, generate_time = adapter_generate_image(
                            f"/kaggle/input/lora-model/lora/{style}",
                            method, speed, lora_name, prompt, negative_prompt)
                    images.append(generate_image)
                    titles.append(f"base + lora + {speed}")
                    descriptions.append(f"excuteTime: {generate_time} s")
            name_1 = lora_name.replace(".safetensors", "")
            merger_file = f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{lora_type}-{name_1}.jpg"
            merger_image(images, f"{lora_type}-{name_1}", titles, descriptions, merger_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Example code for multi-LoRA composition'
    )
    parser.add_argument('--lora_type', default='fuse_save',
                        choices=['fuse_save', 'fuse', 'adapter'],
                        help='methods for combining LoRAs', type=str)
    args = parser.parse_args()
    main(args)



