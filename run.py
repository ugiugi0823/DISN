import gc, pickle, torch, argparse, os, sys
import ptp_utils, seq_aligner

from diffusers import  DiffusionPipeline, DDIMScheduler
from null import NullInversion
from local import AttentionStore, show_cross_attention, run_and_display, make_controller



def main(args):
    
    if args.bigger:
        file_paths = [
            './pickle/x_t_p_1024.pkl',
            './pickle/uncond_embeddings_p_p_1024.pkl',
            './pickle/uncond_embeddings_p_1024.pkl']
        
    else:
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
        
    
    
    prompts = [prompt, prompt]
    controller = AttentionStore()
    neg_prompts =  [neg_prompt, neg_prompt]

    if verbose:
        null_inversion = NullInversion(DISN)
        (image_gt, image_enc), x_t, uncond_embeddings, uncond_embeddings_p = null_inversion.invert(image_path, prompt, verbose=verbose, do_1024=args.bigger)
        torch.cuda.empty_cache()
        gc.collect()

        save_to_pickle(x_t, 'x_t_p', args.bigger)
        save_to_pickle(uncond_embeddings, 'uncond_embeddings_p', args.bigger)
        save_to_pickle(uncond_embeddings_p, 'uncond_embeddings_p_p', args.bigger)
        image_inv, x_t = run_and_display(DISN,neg_prompts,prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, uncond_embeddings_p=uncond_embeddings_p,verbose=False)
        ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
        ptp_utils.save_individual_images([image_gt, image_enc, image_inv[0]])
        show_cross_attention(DISN,prompts,controller, 32, ["up","down","mid"])
    else:
        x_t = load_from_pickle('x_t_p', args.bigger)
        uncond_embeddings = load_from_pickle('uncond_embeddings_p', args.bigger)
        uncond_embeddings_p = load_from_pickle('uncond_embeddings_p_p', args.bigger)



    
    

    ###################################### Various Defect Generation 

    prompts = ["photo of a crack defect image",
            "photo of a crack corrosion image"]
    neg_prompts = [neg_prompt, neg_prompt] 

    cross_replace_steps = {'default_':1.0,}
    self_replace_steps = 0.4
    blend_word = ((('defect',), ("corrosion",))) 
    eq_params = {"words": ("corrosion",), "values": (2,)} # amplify attention to the word "red" by *2

    torch.cuda.empty_cache()
    gc.collect()
    controller = make_controller(DISN,prompts, True, cross_replace_steps, self_replace_steps, blend_word, eq_params, blend_word)
    images, _ = run_and_display(DISN,neg_prompts, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,uncond_embeddings_p=uncond_embeddings_p, steps=50)




if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, default="./img/[0001]TopBF0.png", help="Image Path") 
    p.add_argument("--prompt", type=str, default="photo of a crack defect image", help="Positive Prompt")  
    p.add_argument("--neg_prompt", type=str, default="", help="Negative Prompt")  
    p.add_argument("--bigger", action='store_true', help="If you want to create an image 1024")
  
    args = p.parse_args()

    
    main(args)
  


