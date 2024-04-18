import gc
import abc
import torch
import ptp_utils
import seq_aligner
import shutil

import torch.nn.functional as nnf
import numpy as np

from torch.optim.adam import Adam
from PIL import Image 
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
 
from null import *
from local import *

import torch
import pickle





# CUDA_VISIBLE_DEVICES=2 python run.py

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = "stabilityai/stable-diffusion-xl-base-1.0"
ldm_stable = DiffusionPipeline.from_pretrained(
        model,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    ).to(device)

ldm_stable.load_lora_weights(pjf_path, 
                             weight_name="pytorch_lora_weights.safetensors")

ldm_stable.disable_xformers_memory_efficient_attention()
ldm_stable.enable_model_cpu_offload()

# Train Below
######################################################################################################### 
null_inversion = NullInversion(ldm_stable)
image_path = "./img/[0001]TopBF0.png"
prompt = "photo of a crack defect image"
neg_prompt = ""
(image_gt, image_enc), x_t, uncond_embeddings, uncond_embeddings_p = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True)

print("Modify or remove offsets according to your image!")

# Cleaning memory
######################################################################################################### 
torch.cuda.empty_cache()
gc.collect()

# Infering below
#########################################################################################################

prompts = [prompt, prompt]

controller = AttentionStore()
neg_prompts =  [neg_prompt, neg_prompt]


image_inv, x_t = run_and_display(ldm_stable,neg_prompts,prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, uncond_embeddings_p=uncond_embeddings_p,verbose=False)
print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
ptp_utils.save_individual_images([image_gt, image_enc, image_inv[0]])
show_cross_attention(ldm_stable,prompts,controller, 32, ["up","down"])

prompts = ["photo of a crack defect image", 
           "photo of a crack defect image"]
neg_prompts = [neg_prompt, neg_prompt] 


prompts_make = [re.sub(r"\((.*?)\)", r"\1", prompt) for prompt in prompts]  # 괄호 안의 텍스트 유지
prompts_make = [re.sub(r"\d+\.\d+", "", prompt).strip() for prompt in prompts_make]  # 숫자 제거


cross_replace_steps = {'default_':0.8,}
self_replace_steps = 0.7
blend_word = ((('defect',), ("defect",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
eq_params = {"words": ("defect",), "values": (4.0,)} # amplify attention to the word "red" by *2

controller = make_controller(ldm_stable,prompts_make, True, cross_replace_steps, self_replace_steps, blend_word, eq_params, blend_word)
images, _ = run_and_display(ldm_stable,neg_prompts, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,uncond_embeddings_p=uncond_embeddings_p, steps=50)

print("Image is highly affected by the self_replace_steps, usually 0.4 is a good default value, but you may want to try the range 0.3,0.4,0.5,0.7 ")



######################################################################################################### 





