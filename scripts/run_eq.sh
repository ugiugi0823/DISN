#!/bin/bash
# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Define the two specific cases
cases=(
  "0.5 8.2"
)

# Define the different ch_prompts
ch_prompts=(
  "photo of a crack wear image"
  "photo of a crack dent image"
  "photo of a crack blistering image"
  "photo of a crack peeling image"
  "photo of a crack rust image"
  "photo of a crack contamination image"
  "photo of a crack degradation image"
  "photo of a crack corrosion image"
)


# Iterate over the defined cases
for case in "${cases[@]}"; do
  # Split the case into eq and replace values
  eq=$(echo $case | cut -d' ' -f1)
  replace=$(echo $case | cut -d' ' -f2)
  
  # Iterate over the different ch_prompts
  for ch_prompt in "${ch_prompts[@]}"; do
    # Run the python script with current eq, replace, and ch_prompt values
    python run_eq.py \
    --image_path "./img/[0001]TopBF0.png" \
    --prompt "photo of a crack defect image" \
    --ch_prompt "$ch_prompt" \
    --neg_prompt " " \
    --eq $eq \
    --cross 1.0 \
    --replace $replace \
    --bigger
  done
done