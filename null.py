import numpy as np
import ptp_utils
import seq_aligner
import abc, shutil
import torch
import torch.nn.functional as nnf

from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL, StableDiffusionPipeline, DDIMScheduler
from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from torch.optim.adam import Adam
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType




LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 5.0
MAX_NUM_WORDS = 77

def load_img(image_path, do_1024=False):
    
    image = Image.open(image_path)

    if do_1024:
        if image.size[0] != 1024:
            image = image.resize((1024, 1024)) 
        
            
    else:
        if image.size[0] != 512:
            image = image.resize((512, 512)) 
            
        
    image = np.array(image)
    print("🌊 image resize = ",image.shape)
    if np.isnan(image).any():
        raise ValueError("NaN detected in load_img!")

    return image


class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_sample = self.scheduler.step(model_output, timestep, sample, return_dict=False)[0]
        
        return prev_sample
    
    

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        time_term = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps 
        timestep, next_timestep = min(timestep, 999), min(timestep + time_term, 999)

        
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t        
        beta_prod_t_next = 1 - alpha_prod_t_next

        variance = (beta_prod_t_next / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_next)

        eta = 0.1
        std_dev_t = eta * variance ** (0.5)
        
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5        
        next_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * model_output
        next_sample = alpha_prod_t_next ** (0.5) * next_original_sample + next_sample_direction
        
        
        
        
        return next_sample

    def get_noise_pred_single(self, latents, t, context, context_p, add_time_ids):        
        latents = self.scheduler.scale_model_input(latents, t)
        added_cond_kwargs = {"text_embeds": context_p, "time_ids": add_time_ids}
        noise_pred = self.model.unet(latents, 
                                     t, 
                                     encoder_hidden_states=context,
                                     added_cond_kwargs=added_cond_kwargs,
                                     )["sample"]
        return noise_pred
    

    
    


    def get_noise_pred(self, latents, t, is_forward=True, context=None, context_p=None, add_time_ids=None):
        latents_input = torch.cat([latents] * 2)
        
        context = context if context is not None else self.context
        context_p = context_p if context_p is not None else self.context_p
        add_time_ids = add_time_ids if add_time_ids is not None else self.add_time_ids
        
        
        
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
    
        
        latents_input = self.scheduler.scale_model_input(latents_input, t)
        added_cond_kwargs = {"text_embeds": context_p, "time_ids": add_time_ids}
        noise_pred = self.model.unet(latents_input, 
                                     t, 
                                     encoder_hidden_states=context,
                                     added_cond_kwargs=added_cond_kwargs,
                                     )["sample"]
       
        
        
        
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents
    
    
# 여기는 손을 봤음 
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.13025 * latents.detach()
        # latents = latents.to(torch.float32)
        
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]        
            image = (image * 255).round().astype(np.uint8)
            
            
            
        
        return image




    @torch.no_grad()
    def image2latent(self, image):

        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                
                image = torch.from_numpy(image).float() / 127.5 - 1   
                image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
                
                generator = torch.Generator("cuda").manual_seed(33)

                latents = self.model.vae.encode(image.to(self.model.unet.dtype)).latent_dist.sample(generator)
                # latents = self.model.vae.encode(image.to(self.model.unet.dtype))['latent_dist'].mean
                
                
                if torch.isnan(latents).any():
                    print("wldnjsdfjklsfjkld")
                    # raise ValueError("NaN detected in image2latent!")
                    
                
                latents = latents * 0.13025

                print("🌊 latents.size = ",latents.size())    
        return latents




    @torch.no_grad()
    def init_prompt(self, prompt: str):
        
        compel = Compel(
        tokenizer=[self.model.tokenizer, self.model.tokenizer_2] ,
        text_encoder=[self.model.text_encoder, self.model.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
        )
        
        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel("") 
   
        self.model.vae_scale_factor = 2 ** (len(self.model.vae.config.block_out_channels) - 1)
        self.model.default_sample_size = self.model.unet.config.sample_size
        
        height = self.model.default_sample_size * self.model.vae_scale_factor
        width =  self.model.default_sample_size * self.model.vae_scale_factor
        
        
        
        

        original_size =  (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)    
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        
        passed_add_embed_dim = (
            self.model.unet.config.addition_time_embed_dim * len(add_time_ids) + self.model.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.model.unet.add_embedding.linear_1.in_features

        

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=self.model.unet.dtype).to(self.model.device)
        batch_size = prompt_embeds.shape[0]
        num_images_per_prompt = 1
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        

        
        self.context = torch.cat([negative_prompt_embeds, prompt_embeds])
        self.context_p = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        self.add_time_ids = torch.cat([add_time_ids, add_time_ids])        
        self.prompt = prompt


    @torch.no_grad()
    def ddim_loop(self, latent):

        uncond_embeddings_p, cond_embeddings_p = self.context_p.chunk(2)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        add_time_ids1, add_time_ids2 = self.add_time_ids.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        
        
        
        
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings, cond_embeddings_p, add_time_ids2)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
            
            
        return all_latent


    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
               
        image_rec = self.latent2image(latent)

        ddim_latents = self.ddim_loop(latent)

        return image_rec, ddim_latents
    
    

    def null_optimization(self, latents, num_inner_steps, epsilon):
    
        uncond_embeddings_p, cond_embeddings_p = self.context_p.chunk(2)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        add_time_ids1, add_time_ids2 = self.add_time_ids.chunk(2)
        
            
        uncond_embeddings_list = []
        uncond_embeddings_p_list = []
        add_time_ids1_list = []
        latent_cur = latents[-1]
        # print(latent_cur.size())

        # Set total for tqdm
        # total_iterations = num_inner_steps * NUM_DDIM_STEPS 
        total_iterations = NUM_DDIM_STEPS 
        bar = tqdm(total=total_iterations)
        
        
        for i in range(NUM_DDIM_STEPS):
            
            
            uncond_embeddings = uncond_embeddings.clone().detach().requires_grad_(True)
            uncond_embeddings_p = uncond_embeddings_p.clone().detach().requires_grad_(True)
            add_time_ids1 = add_time_ids1.clone().detach().requires_grad_(True)
            
            optimizer = Adam([uncond_embeddings, uncond_embeddings_p], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings, cond_embeddings_p, add_time_ids2)

            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings, uncond_embeddings_p, add_time_ids1)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                loss_item = loss.item()
                optimizer.zero_grad()
                loss.backward()
                if torch.isnan(uncond_embeddings.grad).any():
                    
                    print("Nan!!")
                    # bar.update()
                    break
                optimizer.step()
                loss_item = loss.item()
                bar.set_description(f"Step {i}, Iteration {j}/{num_inner_steps}, Loss: {loss_item:.4f}")
                
                if loss_item < epsilon + i * 2e-5:
                    break
                
        
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            uncond_embeddings_p_list.append(uncond_embeddings_p[:1].detach())
            
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                context_p = torch.cat([uncond_embeddings_p, cond_embeddings_p])
                add_time_ids = torch.cat([add_time_ids1, add_time_ids2]) 
                latent_cur = self.get_noise_pred(latent_cur, t, False, context, context_p=context_p, add_time_ids=add_time_ids)
            bar.update()
            
        
        return uncond_embeddings_list, uncond_embeddings_p_list


    def invert(self, image_path: str, prompt: str, num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False, do_1024=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_img(image_path, do_1024)
        if verbose:
            print("----- DDIM inversion...")
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        if verbose:
            print("----- Null-text optimization...")
        uncond_embeddings, uncond_embeddings_p = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        # uncond_embeddings = None
        # uncond_embeddings_p = None
        
        
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings, uncond_embeddings_p


    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.add_time_ids = None
        self.context = None
        self.context_p = None

