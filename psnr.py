
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import img_as_ubyte
import cv2  # OpenCV 라이브러리

imageA_path = "/Users/jhw/Desktop/VCL/icpr/ori.png"
imageB_path = "/Users/jhw/Desktop/VCL/icpr/1/dreambooth_aieou.png"

path_parts = imageB_path.split('/')[6]
sub_path = f"/Users/jhw/Desktop/VCL/icpr/{path_parts}" 

imageA = cv2.imread(imageA_path)
imageB = cv2.imread(imageB_path)

# imageA = cv2.resize(imageA, (512, 512))


# RGB 채널별 PSNR 계산
channels = ['B', 'G', 'R']
for i, channel in enumerate(channels):
    original_channel = imageA[:, :, i]
    compared_channel = imageB[:, :, i]
    psnr_value = cv2.PSNR(original_channel, compared_channel)
    # print(f"{channel} 채널의 PSNR 값: {psnr_value}")

# 전체 이미지에 대한 PSNR도 계산할 수 있습니다.
psnr_total = cv2.PSNR(imageA, imageB)
print(f"PSNR: {psnr_total}")



# SSIM 계산
ssim_index, diff = compare_ssim(imageA, imageB, full=True, channel_axis=2,win_size=7)
print(f"SSIM: {ssim_index}")

# diff 이미지를 표시하고 저장하기
full_path_fig = f"{sub_path}/diff_image.png"
plt.figure(figsize=(10, 7))
plt.imshow(diff, cmap='coolwarm')
plt.colorbar()
# plt.title("Image Difference")
plt.xticks([])  # x축의 눈금을 숨깁니다.
plt.yticks([])  # y축의 눈금을 숨깁니다.
plt.savefig(full_path_fig, bbox_inches='tight', pad_inches=0)  # diff 이미지를 파일로 저장
# plt.show()


# diff 이미지를 정규화하고 uint8 타입으로 변환
diff_normalized = (diff * 255).astype(np.uint8)


full_path_cv = f"{sub_path}/diff_image_cv2.png"
cv2.imwrite(full_path_cv, diff_normalized)



full_path_txt = f"{sub_path}/results.txt"
# 결과를 파일에 저장
with open(full_path_txt, 'w') as file:
    file.write(f"비교한 이미지들:\n")
    file.write(f"이미지 A: {imageA_path}\n")
    file.write(f"이미지 B: {imageB_path}\n")
    file.write(f"\n전체 이미지의 PSNR 값: {psnr_total}\n")
    file.write(f"SSIM: {ssim_index}\n")

print("결과가 results.txt 파일에 저장되었습니다.")

