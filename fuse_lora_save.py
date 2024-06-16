import argparse
import os
import torch
from diffusers import DiffusionPipeline, AutoencoderKL, DDIMScheduler, TCDScheduler
from diffusers import LCMScheduler
from huggingface_hub import hf_hub_download
import time
from callbacks import make_callback


def scheduler_save(speed_type: str, pipeline):
    if speed_type == "LCM":
        print("scheduler is >>>>>>>>>> LCM")
        return LCMScheduler.from_config(pipeline.scheduler.config)
    elif speed_type == "Hyper-SD":
        print("scheduler is >>>>>>>>>> Hyper")
        return DDIMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
    else:
        print("scheduler is >>>>>>>>>> TCD")
        return TCDScheduler.from_config(pipeline.scheduler.config)

def speed_choose_save(speed_type: str, pipeline):
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


def generate_image(lora_path: str, method: str, speed_type: str, lora_name: str, prompt, negative_prompt):
    set_vae = False
    if lora_path.find("anime"):
        hug_name = 'gsdf/Counterfeit-V2.5'
    else:
        hug_name = 'SG161222/Realistic_Vision_V5.1_noVAE'
        set_vae = True
    model_name = f'models/{speed_type}'
    if not os.path.exists(model_name):
        # set base model
        pipeline = DiffusionPipeline.from_pretrained(
            hug_name,
            custom_pipeline="./pipelines/sd1.5_0.26.3",
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
        pipeline.scheduler = scheduler_save(speed_type, pipeline)
        speed_choose_save(speed_type, pipeline)
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
    pipeline.load_lora_weights(lora_path, lora_scale=0.8, weight_name=lora_name, adapter_name="lora_style")
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
        num_inference_steps=8,
        guidance_scale=1.8,
        generator=torch.manual_seed(42),
        cross_attention_kwargs={"scale": 0.8},
        callback_on_step_end=switch_callback,
        lora_composite=True if method == "composite" else False
    ).images[0]
    end_time = time.time()
    name_1 = lora_name.replace(".safetensors", "")
    image.save(f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{name_1}-{method}-{speed_type}-loadLora1.jpg")
    return f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{name_1}-{method}-{speed_type}-loadLora1.jpg", int(end_time - start_time)