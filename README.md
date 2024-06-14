## DISN: Deterministic Synthesis of Defect Images using Null Optimization<br><sub>Official PyTorch Implementation</sub>

![Figure 1](./fig/Figure1.jpg)
This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper exploring Deterministic Synthesis of Defect Images using Null Optimization (DISN) 




This repository contains:

* 🪐 A simple PyTorch [implementation](run.sh) of DISN
* ⚡️ Easy Data Augmentation Using our methodology [implementation](run_dataset.sh) 
* 💥 Our LoRA weight [LoRA](./lora/pytorch_lora_weights.safetensors)

## To Do

- [x] LoRA weight upload 
- [x] Create a Dataset Double 
- [x] Making long options in Various Defect Generation 
- [x] Image size 1024 version


## Setup

First, download and set up the repo:
Code has been tested on Cuda 11.8 but other versions should be fine.

```bash
git clone https://github.com/ugiugi0823/DISN.git
cd DISN
```

We provide an environment.yml file that can be used to create a Conda environment. If you only want to run pre-trained models locally on CPU, you can remove the cudatoolkit and pytorch-cuda requirements from the file.
```bash
conda env create -f environment.yaml
conda activate dune
```


## 0. If you want to see the demo like the picture below

| Original | Generated |
|:--------:|:---------:|
| ![Original](./fig/result_0.png) | ![Generated](./fig/result_1.png) |

```bash
bash scripts/run.sh
```
```bash
bash scripts/run_1024.sh
```
It is very similar to the original, but with psnr numbers, you can create a completely different image.
## 1. What if you actually wanted to double up your existing dataset?

```bash
bash scripts/run_dataset.sh

```
```bash
bash scripts/run_dataset_1024.sh

```
If you want to use your dataset, please modify the --original_dataset_path in run_dataset.sh.
Check results.txt later to check PSNR, SSIM, and LPIPS score.


## 2. If you want to see various defect like the picture below

| Original | Corrosion | Degradation |
|:--------:|:---------:| :---------:|
| ![Original](./fig/result_0.png)| ![Corrosion](./fig/corrosion_[0001]TopBF0.png) | ![Degradation](./fig/degradation_[0001]TopBF0.png) |
| Original | Peeling | Wear |
| ![Original](./fig/result_0.png)| ![Peeling](./fig/peeling_[0001]TopBF0.png) | ![wear](./fig/wear_[0001]TopBF0.png) |
```
prompts = ["photo of a crack defect image",
            "photo of a crack corrosion image"]
```

```bash
CUDA_VISIBLE_DEVICES=0 python run_various.py \
--image_path "./img/[0001]TopBF0.png" \
--prompt "photo of a crack defect image" \
--ch_prompt "photo of a crack corrosion image" \
--neg_prompt " " \
```


```bash
bash scripts/run_various.sh
```
```bash
bash scripts/run_various_2024.sh
```

Results of changing text to defect > correlation using existing prompts
It can be confirmed that the defect is corroded compared to the original.



## Acknowledgments
This work was supported by the Institute for Institute of Information \& communications Technology Planning \& Evaluation (IITP) funded by the Ministry of Science and ICT, Government of the Republic of Korea under Project Number RS-2022-00155915. This work was supported by Inha University Research Grant.


<div style="display: flex; justify-content: space-around;">
  <img src="./fig/inha.png" width="10%">
  <img src="./fig/ai_center.png" width="30%">
  <img src="./fig/wta2.png" width="30%">
</div>



## License
The code and model weights are licensed under CC-BY-NC. See [`LICENSE.txt`](LICENSE.txt) for details.