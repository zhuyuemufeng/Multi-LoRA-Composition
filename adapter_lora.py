import torch
from diffusers import DiffusionPipeline, AutoencoderKL, DDIMScheduler, TCDScheduler
from diffusers import LCMScheduler
from huggingface_hub import hf_hub_download
import time
from callbacks import make_callback
def generate_image(lora_path: str, method: str, speed_type: str, lora_name: str, prompt, negative_prompt):
    set_vae = False
    if lora_path.find("anime"):
        model_name = 'gsdf/Counterfeit-V2.5'
    else:
        model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'
        set_vae = True
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="./pipelines/sd1.5_0.26.3",
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")
    if set_vae:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
        ).to("cuda")
        pipeline.vae = vae
    pipeline.scheduler = scheduler(speed_type, pipeline)
    if speed_type == "LCM":
        print("select the speed is >>>>>>>>>> LCM")
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="speed_lora")
    elif speed_type == "Hyper-SD":
        print("select the speed is >>>>>>>>>> Hyper")
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SD15-4steps-lora.safetensors"
        pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name), adapter_name="speed_lora")
    else:
        print("select the speed is >>>>>>>>>> TCD")
        pipeline.scheduler = TCDScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_lora_weights("h1t/TCD-SD15-LoRA", adapter_name="speed_lora")

    # initialize LoRAs
    # This example shows the composition of a character LoRA and a clothing LoRA
    pipeline.load_lora_weights(lora_path, weight_name=lora_name, adapter_name="lora_style", lora_scale=0.8)
    cur_loras = ["speed_lora, lora_style"]

    # select the method for the composition
    if method == "merge":
        pipeline.set_adapters(cur_loras)
        switch_callback = None
    elif method == "switch":
        pipeline.set_adapters([cur_loras[0]])
        switch_callback = make_callback(switch_step=2, loras=cur_loras)
    else:
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
    name_1 = lora_name.replace(".safetensors", "")
    image.save(f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{name_1}-{method}-{speed_type}-loadLora3.jpg")
    return f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{name_1}-{method}-{speed_type}-loadLora3.jpg", int(end_time - start_time)