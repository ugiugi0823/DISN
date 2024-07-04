CUDA_VISIBLE_DEVICES=0 python run_dataset_various.py \
--original_dataset_path "./original_dataset" \
--new_dataset_path "./new_dataset" \
--prompt "photo of a crack defect image" \
--ch_prompt "photo of a crack corrosion image" \
--neg_prompt " " \
--eq 0.5 \
--replace 8.2 \
--datacheck