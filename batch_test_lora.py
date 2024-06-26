import argparse
from image_Util import merger_image, zipDir, merge_images_vertically
from adapter_lora import generate_image as adapter_generate_image
from fuse_lora_current import generate_image as fuse_current_generate_image
from fuse_lora_save import generate_image as fuse_save_generate_image
from base_lora import generate_image as base_generate_image
from utils import load_lora_info, get_prompt


def main(lora_type, lora_method_arg):
    print("start test>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    image_style = ["anime", "reality"]
    # lora_method = ["merge", "switch", "composite"]
    lora_method = [lora_method_arg]
    speed_type = ["LCM", "Hyper-SD", "TCD"]
    for style in image_style:
        print("style>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + style)
        lora_info = load_lora_info(style)
        results = []
        for category in lora_info.values():
            for item in category:
                result = {
                    'id': item['id'],
                    'trigger': item['trigger']
                }
                results.append(result)
        init_prompt, negative_prompt = get_prompt(image_style)
        for lora_data in results:
            lora_name = lora_data["id"] + ".safetensors"
            print("lora_name>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + lora_name)
            lora_tagger = lora_data['trigger']
            prompt = init_prompt + ', ' + ', '.join(lora_tagger)
            print(f"prompt: {prompt}")
            images = []
            titles = []
            descriptions = []
            base_lora_image, base_lora_time = base_generate_image(f"/kaggle/input/lora-model/lora/{style}",
                                                                  lora_name, prompt, negative_prompt, True)
            base_image, base_time = base_generate_image(f"/kaggle/input/lora-model/lora/{style}",
                                                                  lora_name, prompt, negative_prompt, False)
            images.append(base_image)
            titles.append(f"base")
            descriptions.append(f"excuteTime: {base_time} s")
            images.append(base_lora_image)
            titles.append(f"base + lora")
            descriptions.append(f"excuteTime: {base_lora_time} s")
            for method in lora_method:
                print("method>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + method)
                print("lora_type>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + lora_type)
                for speed in speed_type:
                    print("speed>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + speed)
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
                merger_file = f"/kaggle/working/Multi-LoRA-Composition/all_file_image/{style}/all-{method}-{lora_type}-{name_1}.jpg"
                merger_image(images, "", titles, descriptions, merger_file)

                titles.clear()
                images.clear()
                descriptions.clear()
                images.append(base_image)
                titles.append(f"base")
                descriptions.append(f"excuteTime: {base_time} s")
                images.append(base_lora_image)
                titles.append(f"base + lora")
                descriptions.append(f"excuteTime: {base_lora_time} s")
        print("start merge all image>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        merge_images_vertically(f"/kaggle/working/Multi-LoRA-Composition/all_file_image/{style}",
                                f"/kaggle/working/Multi-LoRA-Composition/all_file_image/merge-{style}.jpg")
    zipDir("/kaggle/working/Multi-LoRA-Composition/all_file_image", "/kaggle/working/Multi-LoRA-Composition/all.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Example code for multi-LoRA composition'
    )
    parser.add_argument('--lora_type', default='fuse_save',
                        choices=['fuse_save', 'fuse', 'adapter'],
                        help='methods for combining LoRAs', type=str)
    parser.add_argument('--lora_method', default='switch',
                        choices=["merge", "switch", "composite"],
                        help='methods for combining LoRAs', type=str)
    args = parser.parse_args()
    main(args.lora_type, args.lora_method)



