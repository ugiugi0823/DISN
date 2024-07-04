import gc, pickle, torch, os, argparse, sys, cv2, lpips, time
import ptp_utils, seq_aligner
import numpy as np

from PIL import Image
from tqdm import tqdm
from diffusers import  DiffusionPipeline, DDIMScheduler
from null import NullInversion
from local import AttentionStore, show_cross_attention, run_and_display, make_controller
from torch.cuda import memory_allocated, memory_reserved, max_memory_allocated, max_memory_reserved

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms



def main(args):
    
    
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

    null_inversion = NullInversion(DISN)

    
    image_path = args.original_dataset_path
    directory = args.new_dataset_path
    prompt = args.prompt
    neg_prompt = args.neg_prompt
    
    
    extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [os.path.join(dirpath, filename)
                for dirpath, dirnames, filenames in os.walk(image_path)
                for filename in filenames
                if filename.lower().endswith(extensions)]

    
    
    
    print("🌊 pos_prompt = ", prompt)
    print("🌊 net_prompt = ", neg_prompt)
    prompts = [prompt, prompt]
    neg_prompts =  [neg_prompt, neg_prompt]
    
    bar = tqdm(total=len(image_files))
    
    def find_difference(prompt, ch_prompt):
        
        prompt_words = prompt.split()
        ch_prompt_words = ch_prompt.split()
        
        
        differences = []
        
        
        for word1, word2 in zip(prompt_words, ch_prompt_words):
            if word1 != word2:
                differences.append(word2)
        
        
        if len(differences) != 1:
            raise ValueError("There is not exactly one other word in two sentences.")
        
        return differences[0]
    
    
    
    for i, file_path in enumerate(image_files):   
        start_time = time.time()
        
        start_memory_allocated = memory_allocated()
        start_memory_reserved = memory_reserved()
        start_max_memory_allocated = max_memory_allocated()
        start_max_memory_reserved = max_memory_reserved()
        
        
        
        file_name = file_path.split("/")[-1]
        bar.set_description(f"Creating {i}/{len(image_files)}-th {file_name}")
        (image_gt, image_enc), x_t, uncond_embeddings, uncond_embeddings_p = null_inversion.invert(file_path, prompt, verbose=True, do_1024=args.bigger)  
        torch.cuda.empty_cache()
        gc.collect()
        
        
        


        
        
        cross = 1.0
        eq = args.eq # 2
        replace = args.replace # 0.8
        ch_prompt = args.ch_prompt
        prompts = [prompt, ch_prompt]
        
        pt = find_difference(prompt, ch_prompt)
        print("🌊"*40)
        print("🌊 cross", cross)
        print("🌊 eq", eq)
        print("🌊 replace", replace)
        print("🌊 The word you want to change = ", pt)
        print("🌊"*40)
 
 
        neg_prompts = [neg_prompt, neg_prompt] 
        cross_replace_steps = {'default_':cross,}
        
        blend_word = ((('defect',), (pt,))) 
        eq_params = {"words": (pt,), "values": (eq,)} # amplify attention to the word "corrosion" by *2


        controller = make_controller(DISN,prompts, False, cross_replace_steps, replace, blend_word, eq_params, blend_word)
        images, _ = run_and_display(DISN,neg_prompts, prompts, controller,run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,uncond_embeddings_p=uncond_embeddings_p, steps=50)
        pil_img = Image.fromarray(images[1])
        
        file_name_2 = file_name.split("/")[-1]
        file_name_3 = file_name_2.split(".")[0]
        file_path = f"{directory}/{file_name_3}_{pt}.png"
        pil_img.save(file_path)
        print(f"Image saved as {file_path}") 
        torch.cuda.empty_cache()
        gc.collect()
        
        
        end_time = time.time()
        
        end_memory_allocated = memory_allocated()
        end_memory_reserved = memory_reserved()
        end_max_memory_allocated = max_memory_allocated()
        end_max_memory_reserved = max_memory_reserved()

        elapsed_time = end_time - start_time
        memory_used_allocated = end_memory_allocated - start_memory_allocated
        memory_used_reserved = end_memory_reserved - start_memory_reserved
        peak_memory_allocated = end_max_memory_allocated - start_max_memory_allocated
        peak_memory_reserved = end_max_memory_reserved - start_max_memory_reserved

        
        print(f"Processing {i}/{len(image_files)}-th {file_name}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Memory used (allocated): {memory_used_allocated / (1024 ** 2):.2f} MB")
        print(f"Memory used (reserved): {memory_used_reserved / (1024 ** 2):.2f} MB")
        print(f"Peak memory (allocated): {peak_memory_allocated / (1024 ** 2):.2f} MB")
        print(f"Peak memory (reserved): {peak_memory_reserved / (1024 ** 2):.2f} MB")
        bar.set_description(f"Creating {i}/{len(image_files)}-th {file_name}")
        bar.update()
    
    
    
   
    if args.datacheck:
        
        image_path = args.original_dataset_path
        directory = args.new_dataset_path
        
        original_files = os.listdir(image_path)
        new_files = os.listdir(directory)
        
        results_file = f'{directory}/results.txt'
        with open(results_file, 'w') as file2:
            file2.write('PSNR, SSIM, LPIPS, File\n')

        
        matched_files = [file for file in original_files if f'new_{file}' in new_files]
        
        percept = lpips.LPIPS(net='vgg').cuda()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        
        for file in matched_files:
            print("Check",file)
            
            original_image = Image.open(os.path.join(image_path, file))
            original_image = original_image.resize((512, 512))
            new_image = Image.open(os.path.join(directory, f'new_{file}'))
            
            original_image = np.array(original_image)
            new_image = np.array(new_image)
            
            
            imageA_t = transform(original_image).unsqueeze(0).cuda()
            imageB_t = transform(new_image).unsqueeze(0).cuda()
            
            # PSNR 
            psnr_value = psnr(original_image, new_image)
            # SSIM 
            ssim_value, _ = ssim(original_image, new_image, full=True, channel_axis=2, win_size=7)
            # LPIPS
            lpips_value = percept(imageA_t, imageB_t).item()
            with open(results_file, 'a') as file2:
                
                file2.write(f'{psnr_value:.2f}, {ssim_value:.4f}, {lpips_value:.4f}, {file}\n')
                
        print("🌊 Check out all the results in results.txt!")

  
    
    



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--original_dataset_path", type=str, default="./original_dataset", help="Existing Data Paths") 
    p.add_argument("--new_dataset_path", type=str, default="./new_dataset", help="The path of the new image to be saved")  
    p.add_argument("--prompt", type=str, default="photo of a crack defect image", help="Positive Prompt")  
    p.add_argument("--ch_prompt", type=str, default="photo of a crack corrosion image", help="Change Prompt")  
    p.add_argument("--neg_prompt", type=str, default="", help="Negative Prompt")  
    p.add_argument("--eq", type=float, default="0.5", help="eq value")  
    p.add_argument("--replace", type=float, default="8.2", help="replace value")
    p.add_argument("--bigger", action='store_true', help="If you want to create an image 1024")
    p.add_argument("--datacheck", action='store_true', help="If you want to compare the generated image with the original image")
  
  
  
    args = p.parse_args()
    
    if not os.path.exists(args.original_dataset_path):
        print(f"Error: The directory '{args.original_dataset_path}' does not exist.")
        sys.exit(1)  
        
    if not os.path.exists(args.new_dataset_path):
        os.makedirs(args.new_dataset_path, exist_ok=True)
        print(f"Directory '{args.new_dataset_path}' was created.")
    
    main(args)
  





