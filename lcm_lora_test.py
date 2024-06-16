import argparse
import os
import torch
from diffusers import DiffusionPipeline, AutoencoderKL, DDIMScheduler, TCDScheduler
from diffusers import LCMScheduler
from huggingface_hub import hf_hub_download
import time
from callbacks import make_callback
from PIL import Image, ImageDraw, ImageFont

def get_example_prompt():
    prompt = "RAW photo, subject, 8k uhd, dslr, high quality, Fujifilm XT3, half-length portrait from knees up, scarlett, short red hair, blue eyes"
    negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    return prompt, negative_prompt


def scheduler(speed_type: str, pipeline):
    if speed_type == "LCM":
        print("scheduler is >>>>>>>>>> LCM")
        return LCMScheduler.from_config(pipeline.scheduler.config)
    elif speed_type == "Hyper-SD":
        print("scheduler is >>>>>>>>>> Hyper")
        return DDIMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
    else:
        print("scheduler is >>>>>>>>>> TCD")
        return TCDScheduler.from_config(pipeline.scheduler.config)

def speed_choose(speed_type: str, pipeline):
    if speed_type == "LCM":
        print("speed_choose is >>>>>>>>>> LCM")
        pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    elif speed_type == "Hyper-SD":
        print("speed_choose is >>>>>>>>>> Hyper")
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SD15-4steps-lora.safetensors"
        pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    else:
        print("speed_choose is >>>>>>>>>> TCD")
        pipeline.load_lora_weights("h1t/TCD-SD15-LoRA")


def merge_images_with_text(images, titles, descriptions, output_path):
    # Load images
    imgs = [Image.open(img_path) for img_path in images]

    # Determine the total width and the max height
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights) + 80  # 80 pixels for title and description

    # Create a new image with a white background
    new_img = Image.new('RGB', (total_width, max_height), 'white')

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # Draw images, titles, and descriptions
    draw = ImageDraw.Draw(new_img)
    x_offset = 0
    for idx, img in enumerate(imgs):
        new_img.paste(img, (x_offset, 40))
        draw.text((x_offset + img.width // 2, 10), titles[idx], fill="black", font=font, anchor="mm")
        draw.text((x_offset + img.width // 2, img.height + 50), descriptions[idx], fill="black", font=font, anchor="mm")
        x_offset += img.width

    # Save the final image
    new_img.save(output_path)

def speed_lora(lora_path: str, method: str, speed_type: str, lora_name: str, bath_fix: str = "1"):
    prompt, negative_prompt = get_example_prompt()
    model_name = f'models/{speed_type}-{bath_fix}'
    if not os.path.exists(model_name):
        # set base model
        pipeline = DiffusionPipeline.from_pretrained(
            'SG161222/Realistic_Vision_V5.1_noVAE',
            custom_pipeline="./pipelines/sd1.5_0.26.3",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        ).to("cuda")

        #set vae
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
        ).to("cuda")
        pipeline.vae = vae
        pipeline.scheduler = scheduler(speed_type, pipeline)
        speed_choose(speed_type, pipeline)
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()
        pipeline.save_pretrained(model_name)

    # set base model
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="./pipelines/sd1.5_0.26.3",
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")
    # set scheduler
    pipeline.scheduler = scheduler(speed_type, pipeline)

    # initialize LoRAs
    # This example shows the composition of a character LoRA and a clothing LoRA
    pipeline.load_lora_weights(lora_path, weight_name=lora_name, adapter_name="lora_style")
    cur_loras = ["lora_style"]

    # select the method for the composition
    if method == "merge":
        print("select the method is >>>>>>>>>> merge")
        pipeline.set_adapters(cur_loras)
        switch_callback = None
    elif method == "switch":
        print("select the method is >>>>>>>>>> switch")
        pipeline.set_adapters([cur_loras[0]])
        switch_callback = make_callback(switch_step=2, loras=cur_loras)
    else:
        print("select the method is >>>>>>>>>> composite")
        pipeline.set_adapters(cur_loras)
        switch_callback = None
    start_time = time.time()
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        # height=1024,
        # width=768,
        num_inference_steps=8,
        guidance_scale=1.8,
        generator=torch.manual_seed(42),
        cross_attention_kwargs={"scale": 0.8},
        callback_on_step_end=switch_callback,
        lora_composite=True if method == "composite" else False
    ).images[0]
    end_time = time.time()
    image.save(f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{method}-{speed_type}-{bath_fix}.jpg")
    return f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{method}-{speed_type}-{bath_fix}.jpg", end_time - start_time

def base_lora(lora_path: str, lora_name: str, bath_fix: str = "1"):
    prompt, negative_prompt = get_example_prompt()
    # set base model
    pipeline = DiffusionPipeline.from_pretrained(
        'SG161222/Realistic_Vision_V5.1_noVAE',
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")
    if lora_name != '':
        pipeline.load_lora_weights(lora_path, weight_name=lora_name, adapter_name="lora_style")
        cur_loras = ["lora_style"]
        pipeline.set_adapters(cur_loras)
    start_time = time.time()
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        # height=1024,
        # width=768,
        num_inference_steps=20,
        guidance_scale=1.8,
        generator=torch.manual_seed(42),
        cross_attention_kwargs={"scale": 0.8},
        lora_composite=False
    ).images[0]
    end_time = time.time()
    file_name = "lora" if lora_name != '' else "no_lora"
    pipeline.unload_lora_weights()
    image.save(f"/kaggle/working/Multi-LoRA-Composition/test_file_image/base-{file_name}-{bath_fix}.jpg")
    return f"/kaggle/working/Multi-LoRA-Composition/test_file_image/base-{file_name}-{bath_fix}.jpg", end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Example code for multi-LoRA composition'
    )

    parser.add_argument('--method', default='switch',
                        choices=['merge', 'switch', 'composite'],
                        help='methods for combining LoRAs', type=str)
    parser.add_argument('--lora', default='clothing_1.safetensors',
                        help='lora name', type=str)
    args = parser.parse_args()


    path = "/kaggle/input/lora-model/lora/reality"
    name = args.lora
    method = args.method   # merge switch composite
    print(f"lora_name: {name}, method: {method}")
    lcm_file, lcm_time = speed_lora(path, method, "LCM", name)

    hyper_file, hyper_time = speed_lora(path, method, "Hyper-SD", name)

    tcd_file, tcd_time = speed_lora(path, method, "TCD", name)

    base_lora_file, base_lora_time = base_lora(path, name)

    base_file, base_time = base_lora("", "")

    # 图片路径列表
    images = [base_file, base_lora_file, lcm_file, hyper_file, tcd_file]

    # 每张图片的标题
    titles = ['Base', 'Base + LORA', 'Base + LORA + LCM', 'Base + LORA + Hyper-SD', 'Base + LORA + TCD']

    # 每张图片的说明文字
    subtitles = [f'execution time: {base_time}', f'execution time: {base_lora_time}',
                 f'execution time: {lcm_time}', f'execution time: {hyper_time}',
                 f'execution time: {tcd_time}']
    name_1 = name.replace(".safetensors", "")
    merge_images_with_text(images, titles, subtitles, f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{method}-{name_1}.jpg")
    print(f"{method}-{name_1} finished")