import gc, pickle, torch, argparse, os, sys
import ptp_utils, seq_aligner
import numpy as np

from PIL import Image
from diffusers import  DiffusionPipeline, DDIMScheduler
from null import NullInversion
from local import AttentionStore, show_cross_attention, run_and_display, make_controller




# CUDA_VISIBLE_DEVICES=4 python run_various.py





def main(args):
    
    file_paths = [
    './pickle/x_t_p.pkl',
    './pickle/uncond_embeddings_p_p.pkl',
    './pickle/uncond_embeddings_p.pkl']
    
    all_files_exist = all(os.path.exists(path) for path in file_paths)

    if all_files_exist:
        content_is_none = False
        for path in file_paths:
            with open(path, 'rb') as file:
                content = pickle.load(file)
                if content is None:
                    content_is_none = True
                    break

        if content_is_none:
            print("🌊 One or more files are empty. Switching to verbose mode.")
            verbose = True
        else:
            print("🌊 All files exist and are non-empty")
            verbose = False
    else:
        print("🌊 All files are missing. Let's start the creation process.")
        verbose = True
    
    
    
    prompt = args.prompt
    neg_prompt = args.neg_prompt
    image_path = args.image_path
    

    ###################################### DISN
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    DISN = DiffusionPipeline.from_pretrained(
            model,
            scheduler=scheduler,
            torch_dtype=torch.float32,
        ).to(device)
    
    
    pjf_path = "./lora"
    DISN.load_lora_weights(pjf_path, weight_name="pytorch_lora_weights.safetensors")
    DISN.disable_xformers_memory_efficient_attention()
    DISN.enable_model_cpu_offload()

    ###################################### DISN
    def get_pickle_path(filename, bigger):
        return f'./pickle/{filename}_1024.pkl' if bigger else f'./pickle/{filename}.pkl'

    def save_to_pickle(data, filename, bigger):
        with open(get_pickle_path(filename, bigger), 'wb') as f:
            pickle.dump(data, f)

    def load_from_pickle(filename, bigger):
        with open(get_pickle_path(filename, bigger), 'rb') as f:
            return pickle.load(f)

    if verbose:
        null_inversion = NullInversion(DISN)
        (image_gt, image_enc), x_t, uncond_embeddings, uncond_embeddings_p = null_inversion.invert(image_path, prompt, verbose=verbose, do_1024=args.bigger)
        torch.cuda.empty_cache()
        gc.collect()

        save_to_pickle(x_t, 'x_t_p', args.bigger)
        save_to_pickle(uncond_embeddings, 'uncond_embeddings_p', args.bigger)
        save_to_pickle(uncond_embeddings_p, 'uncond_embeddings_p_p', args.bigger)
    else:
        x_t = load_from_pickle('x_t_p', args.bigger)
        uncond_embeddings = load_from_pickle('uncond_embeddings_p', args.bigger)
        uncond_embeddings_p = load_from_pickle('uncond_embeddings_p_p', args.bigger)

    

            
        
    ###################################### Various Defect Generation   
    
    def find_difference(prompt, ch_prompt):
        
        prompt_words = prompt.split()
        ch_prompt_words = ch_prompt.split()
        
        
        differences = []
        
        
        max_len = max(len(prompt_words), len(ch_prompt_words))
        prompt_words += [""] * (max_len - len(prompt_words))
        ch_prompt_words += [""] * (max_len - len(ch_prompt_words))
        
        
        for word1, word2 in zip(prompt_words, ch_prompt_words):
            if word1 != word2:
                differences.append(word2)
    
        return differences[0]
    
    
    
    
    eq = 2
    cross = 1.0
    replace = 0.8
    ch_prompt = args.ch_prompt
    prompts = [prompt, ch_prompt]
    
    pt = find_difference(prompt, ch_prompt)
    # pt = "defect"
    print("🌊 ", prompt)
    print("🌊 ", ch_prompt)
    print("🌊 The word you want to change = ", pt)
    
    

    prompts = [prompt,
            f"photo of a crack {pt} image"]
    neg_prompts = [neg_prompt, neg_prompt] 
    
    cross_replace_steps = {'default_':cross,}
    blend_word = ((('defect',), (pt,))) 
    eq_params = {"words": (pt,), "values": (eq,)} # amplify attention to the word "dent" by *2

    torch.cuda.empty_cache()
    gc.collect()
    controller = make_controller(DISN,prompts, False, cross_replace_steps, replace, blend_word, eq_params, blend_word)
    images, _ = run_and_display(DISN,neg_prompts, prompts, controller,run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,uncond_embeddings_p=uncond_embeddings_p, steps=50)
    
    
    ptp_utils.diff_individual([images[0],images[1]])
    pil_img = Image.fromarray(images[1])
    file_path = f"./result/{pt}_{neg_prompt}_cross-{cross}_replace-{replace}_eq-{eq}-2.png"
    pil_img.save(file_path)
    print(f"Image saved as {file_path}") 

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, default="./img/[0001]TopBF0.png", help="Image Path") 
    p.add_argument("--prompt", type=str, default="photo of a crack defect image", help="Positive Prompt")  
    p.add_argument("--ch_prompt", type=str, default="photo of a crack corrosion image", help="Change Prompt")  
    p.add_argument("--neg_prompt", type=str, default="worst quality", help="Negative Prompt")  
    p.add_argument("--bigger", action='store_true', help="If you want to create an image 1024")
  
    args = p.parse_args()
    
    
    pickle_path = "./pickle"  
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path, exist_ok=True)
        print(f"Directory '{pickle_path}' was created.")

    
    main(args)
  


