import torch
from diffusers import DiffusionPipeline, AutoencoderKL
import time

from callbacks import make_callback


def generate_image(lora_type: str, lora_list: list, method: str, prompt, negative_prompt, add_lora: bool = True):
    set_vae = False
    if "anime" == lora_type:
        hug_name = 'gsdf/Counterfeit-V2.5'
    else:
        hug_name = 'SG161222/Realistic_Vision_V5.1_noVAE'
        set_vae = True
    print("Base model: hug_name>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + hug_name)
    pipeline = DiffusionPipeline.from_pretrained(
        hug_name,
        custom_pipeline="./pipelines/sd1.5_0.26.3",
        use_safetensors=True
    ).to("cuda")
    if set_vae:
        # set vae
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
        ).to("cuda")
        pipeline.vae = vae
    switch_callback = None
    if add_lora:
        cur_loras = []
        for lora in lora_list:
            print(f"/kaggle/input/lora-model/lora/{lora_type}/{lora}")
            pipeline.load_lora_weights(f"/kaggle/input/lora-model/lora/{lora_type}/{lora}", adapter_name=lora)
            cur_loras.append(lora)
        print("Base model: add_lora>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> True")
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
        num_inference_steps=20,
        guidance_scale=1.8,
        generator=torch.manual_seed(42),
        cross_attention_kwargs={"scale": 0.8},
        callback_on_step_end=switch_callback,
        lora_composite=True if method == "composite" else False
    ).images[0]
    end_time = time.time()
    name_lora = [lo.replace(".safetensors", "") for lo in lora_list]
    name_1 = ",".join(name_lora)
    if add_lora:
        name_2 = "lora"
    else:
        name_2 = "no_lora"
    image.save(
        f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{lora_type}-{method}-{name_1}-{method}-{name_2}-base.jpg")
    pipeline.unload_lora_weights()
    pipeline.disable_lora()
    return (
    f"/kaggle/working/Multi-LoRA-Composition/test_file_image/{lora_type}-{method}-{name_1}-{method}-{name_2}-base.jpg",
    "{:.2f}".format(end_time - start_time))
