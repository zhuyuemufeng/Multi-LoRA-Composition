import torch
from diffusers import DiffusionPipeline, AutoencoderKL
import time


def generate_image(lora_path: str, lora_name: str, prompt, negative_prompt):
    set_vae = False
    if lora_path.find("anime"):
        hug_name = 'gsdf/Counterfeit-V2.5'
    else:
        hug_name = 'SG161222/Realistic_Vision_V5.1_noVAE'
        set_vae = True
    pipeline = DiffusionPipeline.from_pretrained(
        hug_name,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")
    if set_vae:
        # set vae
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
        ).to("cuda")
        pipeline.vae = vae
    if lora_name != '':
        pipeline.load_lora_weights(f"{lora_path}/{lora_name}", adapter_name="lora_style", lora_scale=0.8)
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
    name_1 = lora_name.replace(".safetensors", "")
    image.save(f"/kaggle/working/Multi-LoRA-Composition/test_file_image/base-{file_name}-{name_1}.jpg")
    pipeline.unload_lora_weights()
    return f"/kaggle/working/Multi-LoRA-Composition/test_file_image/base-{file_name}-{name_1}.jpg", int(end_time - start_time)
