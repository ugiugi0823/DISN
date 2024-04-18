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


######################################################################################################### 





