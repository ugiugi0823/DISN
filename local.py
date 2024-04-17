from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
import torch
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image, ImageEnhance
from typing import Optional, Union, Tuple, List, Callable, Dict

from compel import Compel, ReturnedEmbeddingsType
import re


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
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
 
from null import *
from local import *


# CUDA_VISIBLE_DEVICES=5 python run.py

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")


model = "stabilityai/stable-diffusion-xl-base-1.0"

ldm_stable = DiffusionPipeline.from_pretrained(
        model,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    ).to(device)


pjf_path = "./lora"
ldm_stable.load_lora_weights(pjf_path, 
                             weight_name="pytorch_lora_weights.safetensors") 

ldm_stable.disable_xformers_memory_efficient_attention()
ldm_stable.enable_model_cpu_offload()

tokenizer = ldm_stable.tokenizer











LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 5.0
MAX_NUM_WORDS = 77


class LocalBlend:

    def get_mask(self, x_t, maps, alpha, use_pool):
        k = 1
        # x_t : torch.Size([2, 4, 64, 64]) maps: torch.Size([2, 1280, 1, 16, 16, 77]) alpha: torch.Size([2, 1, 1, 1, 1, 77])
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        wxxk_test = attention_store
        
        
        if self.counter > self.start_blend:
            print("🦖"*2)
            # maps = attention_store["down_cross"][0:2] + attention_store["up_cross"][:3]
            # 🦖1
            # maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            # 🦖2
            maps = attention_store["down_cross"][12:24] + attention_store["up_cross"][:18]
            # 🦖3 
            # maps = attention_store["down_cross"][:1] + attention_store["up_cross"][:1]
            # 🦖4
            # maps = attention_store["down_cross"][:7] + attention_store["up_cross"][:7]
            # 🦖5            
            # maps = attention_store["down_cross"][:24] + attention_store["up_cross"][:36]
            
            # maps = attention_store["down_cross"][22:24] + attention_store["up_cross"][:3]
            # maps = attention_store["down_cross"][:24] + attention_store["up_cross"][:36]
            # maps = attention_store["down_cross"][0:2] + attention_store["up_cross"][:3]
            
            
            maps_list = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps_list, dim=1)
            # maps = maps[:1]
            # 여기 건드림 03.30일   
            mask = self.get_mask(x_t,maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, model,prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        tokenizer = model.tokenizer
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th=th




class EmptyControl:


    def step_callback(self, x_t):
        
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        # 🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥🫥
        
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # attn = attn.reshape(20, 1024, 64)
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        print("🪔"*50)

        key_info = self.attention_store.keys()
        print(key_info)
        print("🪔"*80)
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        
        # average_attention = {
        #     key: [item / self.cur_step for item in self.attention_store[key]]
        #     for key in self.attention_store if self.attention_store[key]
        # }

        
        # average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        # print("class AttentionControlEdit(AttentionStore, abc.ABC):")
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        
        if att_replace.shape[2] <= 32 ** 2:
            
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        
        # self = <class 'local.AttentionReweight'>
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            
            # self.num_self_replace = (0, 35)
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                # print(self.cross_replace_alpha.size())
                # print(self.cur_step)
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(prompts,attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    
    num_pixels = res ** 2
    for location in from_where:
        print("is_cross의 결과는 ~~~~", is_cross)
        print(f"{location}_{'cross' if is_cross else 'self'}")
        if attention_maps[f"{location}_{'cross' if is_cross else 'self'}"] is not None:
            
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                
                # print("🎧")
                # print(item.shape)
                # torch.Size([1, 20, 256, 64])
                # torch.Size([1, 20, 1024, 64])
                # print(item.shape[2])
                
                #torch.Size([20, 256, 77])
                # torch.Size([20, 16, 16, 77])
                # item = item.squeeze(0)
                
                if item.shape[1] == num_pixels:
                    
                    # item.squeeze(0)
                    # print(item.size())
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    # print(cross_maps.size())
                    out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(model,prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None, blend_word=None) -> AttentionControlEdit:
    if blend_words is None:
        print("1️⃣"*40)
        lb = None
    else:
        print("2️⃣"*40)
        print(prompts)
        print(blend_word)
        lb = LocalBlend(model,prompts, blend_word)
    if is_replace_controller:
        print("3️⃣"*40)
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        print("34⃣"*40)
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(model, prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    # 괄호를 제거하지만 괄호 안의 내용은 유지하고, 뒤따르는 숫자를 제거
    prompts = [re.sub(r"\((.*?)\)", r"\1", prompt) for prompt in prompts]  # 괄호 안의 텍스트 유지
    prompts = [re.sub(r"\d+\.\d+", "", prompt).strip() for prompt in prompts]  # 숫자 제거
    
    tokenizer = model.tokenizer 
    tokens = tokenizer.encode(prompts[select])
    
    
    decoder = tokenizer.decode
    # print(f"'from_where' 인자 값: {from_where}")
    # try:
    #     attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    # except KeyError as e:
    #     print(f"주의: {e} 키를 찾을 수 없습니다.")
    #     return  # 혹은 적절한 기본값으로 계속 진행
    
    
    
    attention_maps = aggregate_attention(prompts,attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        
        # Pillow 이미지로 변환
        pil_image = Image.fromarray(image)
        # 명암 대비를 증가시키기
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = enhancer.enhance(4.0)  # 대비 인자 조정
        
        # 256x256 크기로 조정
        resized_image = enhanced_image.resize((256, 256))
        image_with_text = ptp_utils.text_under_image(np.array(resized_image), decoder(int(tokens[i])))
        images.append(image_with_text)
        
        
        # image = np.array(Image.fromarray(image).resize((256, 256)))
        # image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        # images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))
    
    

@torch.no_grad()
def text2image_ldm_stable(
    model,
    neg_prompt:  List[str],
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    uncond_embeddings_p=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    
    compel = Compel(
    tokenizer=[model.tokenizer, model.tokenizer_2] ,
    text_encoder=[model.text_encoder, model.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True]
    )
    

    
    
    
    prompt_embeds, pooled_prompt_embeds = compel(prompt)
    negative_prompt_embeds, negative_pooled_prompt_embeds = compel(neg_prompt) 
    
    
    
###################################🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡
    model.vae_scale_factor = 2 ** (len(model.vae.config.block_out_channels) - 1)
    model.default_sample_size = model.unet.config.sample_size
    
    height = model.default_sample_size * model.vae_scale_factor
    width =  model.default_sample_size * model.vae_scale_factor
    
    # height = width = 512

    original_size =  (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)    
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
    passed_add_embed_dim = (
        model.unet.config.addition_time_embed_dim * len(add_time_ids) + model.text_encoder_2.config.projection_dim
    )
    expected_add_embed_dim = model.unet.add_embedding.linear_1.in_features

    

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=model.unet.dtype).to(model.device)
    batch_size = prompt_embeds.shape[0]
    num_images_per_prompt = 1
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

    
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0) 
    # context_p = torch.cat([pooled_prompt_embeds, negative_pooled_prompt_embeds], dim=0)
    
###################################🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡🫡 
    

    
    # max_length = text_inputs.input_ids.shape[-1]
    if uncond_embeddings is None:

        uncond_embeddings_ = negative_prompt_embeds

    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    # start_time = num_inference_steps
    torch.cuda.empty_cache()
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        # print(i)
        if uncond_embeddings_ is None:
            
            context = torch.cat([uncond_embeddings[i].expand(*prompt_embeds.shape).to(model.device), prompt_embeds.to(model.device)]) 
            context_p = torch.cat([uncond_embeddings_p[i].expand(*pooled_prompt_embeds.shape).to(model.device), pooled_prompt_embeds.to(model.device)]) 
            
            # add_time_idss = torch.cat([add_time_ids1[i].expand(*add_time_ids.shape).to(model.device), add_time_ids.to(model.device)]) 
            
            # context = torch.cat([prompt_embeds, prompt_embeds])
            # context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, prompt_embeds]) 
            print("👑"*40)
        
        # print("🎰"*40)
        
        latents = ptp_utils.diffusion_step(model, controller, latents, context, context_p, add_time_ids, t, guidance_scale, low_resource=False)

    if return_type == 'image':
        print("🤯"*40)
        image = ptp_utils.latent2image(model.vae, latents)
        
    else:
        image = latents
    return image, latent



def run_and_display(model,neg_prompts,prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None,uncond_embeddings_p=None,add_time_ids1=None ,verbose=True, steps=50):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(model, neg_prompts,prompts, controller, latent=latent, num_inference_steps=steps, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, uncond_embeddings_p=uncond_embeddings_p)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t
