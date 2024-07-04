CUDA_VISIBLE_DEVICES=0 python run_various.py \
--image_path "./img/[0001]TopBF0.png" \
--prompt "photo of a crack defect image" \
--ch_prompt "photo of a crack corrosion image" \
--neg_prompt " " \
--eq 0.5 \
--replace 8.2
