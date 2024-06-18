import argparse
import json

from JsonParse import JsonParse
from image_Util import merger_image, zipDir, merge_images_vertically
from adapter_lora_muli import generate_image as adapter_generate_image
from fuse_lora_save_muli import generate_image as fuse_save_generate_image
from base_lora_muli import generate_image as base_generate_image
from utils import load_lora_info, get_prompt


def main(lora_type, lora_method_arg):
    lora_method = [lora_method_arg]
    speed_type = ["LCM", "Hyper-SD", "TCD"]
    with open("image_info_test.json") as f:
        lora_info = json.loads(f.read())
    results = []
    for category in lora_info:
        results.append(JsonParse(category['path'], category['prompt']))
    for json_data in results:
        print("lora_name>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + ",".join(json_data.get_lora_list()))
        print("lora_type>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + json_data.get_lora_type())
        init_prompt, negative_prompt = get_prompt(json_data.get_lora_type())
        prompt = init_prompt + ', ' + json_data.get_taggers()
        for method in lora_method:
            print("method>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + method)
            print("lora_type>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + lora_type)
            images = []
            titles = []
            descriptions = []
            base_lora_image, base_lora_time = base_generate_image(json_data.lora_type, json_data.lora_list, method,
                                                                  prompt, negative_prompt, True)
            base_image, base_time = base_generate_image(json_data.lora_type, json_data.lora_list, method, prompt,
                                                        negative_prompt, False)
            images.append(base_image)
            titles.append(f"base")
            descriptions.append(f"excuteTime: {base_time} s")
            images.append(base_lora_image)
            titles.append(f"base + lora")
            descriptions.append(f"excuteTime: {base_lora_time} s")
            for speed in speed_type:
                print("speed>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + speed)
                if lora_type == "fuse_save":
                    generate_image, generate_time = fuse_save_generate_image(
                        json_data.lora_type, json_data.lora_list,
                        method, speed, prompt, negative_prompt)
                else:
                    generate_image, generate_time = adapter_generate_image(
                        json_data.lora_type, json_data.lora_list,
                        method, speed, prompt, negative_prompt)
                images.append(generate_image)
                titles.append(f"base + lora + {speed}")
                descriptions.append(f"excuteTime: {generate_time} s")
            name_lora = [lo.replace(".safetensors", "") for lo in json_data.get_lora_list()]
            name_1 = ",".join(name_lora)
            merger_file = f"/kaggle/working/Multi-LoRA-Composition/all_file_image/{lora_method_arg}/all-{json_data.lora_type}-{name_1}.jpg"
            merger_image(images, "", titles, descriptions, merger_file)
    print("start merge all image>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    merge_images_vertically(f"/kaggle/working/Multi-LoRA-Composition/all_file_image/{lora_method_arg}",
                            f"/kaggle/working/Multi-LoRA-Composition/all_file_image/merge-muli-{lora_method_arg}.jpg")
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



