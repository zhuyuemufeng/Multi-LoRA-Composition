# Multi-LoRA Composition
Official repository for the paper
*Multi-LoRA Composition for Image Generation*

## Overview

Low-Rank Adaptation (LoRA) is extensively utilized in text-to-image models for the accurate rendition of specific elements like distinct characters or unique styles in generated images.

In this project, we aim to integrate any number of elements in an image by composing multiple LoRAs, and propose two training-free methods to implement the composition, including LoRA Switch and LoRA Composite.

The following figure compares the prevalent LoRA Merge method with our approaches.

<p align="center">
    <img src="images/intro_fig.png" width="100%" alt="intro_case">
</p>

## Preparation

### Environment
To get started, install the necessary packages:
```
conda create --name multi-lora python=3.10
conda activate multi-lora
pip install -r requirements.txt
```

### Download Pre-trained LoRAs
Our **ComposLoRA** benchmark collects 22 pre-trained LoRAs, including character, color, style, background and object. Please download and move `ComposLoRA.zip` to the `models` folder at [this link](https://drive.google.com/file/d/1SuwRgV1LtEud8dfjftnw-zxBMgzSCwIT/view?usp=sharing) and unzip it.

## Image Generation with Multi-LoRA Composition

The following shows how multiple LoRAs can be composed in different methods during image generation.

First load the base model for image generation:

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
        'SG161222/Realistic_Vision_V5.1_noVAE',
        custom_pipeline="MingZhong/StableDiffusionPipeline-with-LoRA-C",
        use_safetensors=True
    ).to("cuda")
```

Here we select a model from huggingface to generate realistic style generation. Our custom pipeline adds an implementation of the LoRA composiste method to the standard stable diffusion pipeline.

Next we select a character LoRA and a clothing LoRA from ComposLoRA for composition.

```python
# Load LoRAs
lora_path = 'models/lora/reality'
pipeline.load_lora_weights(lora_path, weight_name="character_2.safetensors", adapter_name="character")
pipeline.load_lora_weights(lora_path, weight_name="clothing_2.safetensors", adapter_name="clothing")

# List of LoRAs to be composed
cur_loras = ["character", "clothing"]
```

Then select a specific method for composing multiple LoRAs." merge" is the previous method, while "switch" and "composite" are our new proposed approaches.
```python
from callbacks import make_callback

method = 'switch'

# Initialization based on the selected method
if method == "merge":
    pipeline.set_adapters(cur_loras)
    switch_callback = None
elif method == "switch":
    pipeline.set_adapters([cur_loras[0]])
    switch_callback = make_callback(switch_step=args.switch_step, 
                                    loras=cur_loras)
else:
    pipeline.set_adapters(cur_loras)
    switch_callback = None
```
Finally, we set the prompt and generate the image.

```python
# set the prompts for image generation
prompt = "RAW photo, subject, 8k uhd, dslr, high quality, Fujifilm XT3, half-length portrait from knees up, scarlett, short red hair, blue eyes, school uniform, white shirt, red tie, blue pleated microskirt"
negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

# generate and save the image
generator = torch.maunal_seed(11)
image = pipeline(prompt=prompt, 
                 negative_prompt=negative_prompt,
                 height=1024,
                 width=768,
                 num_inference_steps=100,
                 guidance_scale=7,
                 generator=generator,
                 cross_attention_kwargs={"scale": 0.8},
                 callback_on_step_end=switch_callback,
                 lora_composite=True if method == "composite" else False
            ).images[0]
image.save('example.png')
```

The full code is provided in example.py, and you can get the results of the different methods by modifying the following commands:
```
python example.py --method switch
```

The images generated by the three composition methods are:

<div style="text-align: center;">
  <img src="images/merge_example.png" alt="merge_example" style="width: auto; max-width: 20%; margin-right: 50px;">
  <img src="images/switch_example.png" alt="switch_example" style="width: auto; max-width: 20%; margin-right: 50px;">
  <img src="images/composite_example.png" alt="composite_example" style="width: auto; max-width: 20%;">
</div>








