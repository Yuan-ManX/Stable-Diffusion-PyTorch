from ddpm import DDPMSampler

from PIL import Image
import torch
import numpy as np
import math
import matplotlib.pyplot as plt


"""
详细步骤：
1.随机数生成器的初始化：
使用 torch.Generator() 创建生成器，用于控制随机性。
固定随机种子 (manual_seed) 是为了确保在多次运行中获得一致的结果。

2.噪声级别列表：
noise_levels 表示扩散过程中的时间步，数值越高表示加入的噪声越多。
时间步数对应累积噪声量。

3.图像预处理：
图像加载并转为 NumPy 数组。
将像素值从 [0, 255] 转换为 [-1, 1]，以符合扩散模型的输入格式。

4.批量创建：
将单张图像扩展为与 noise_levels 相同数量的批次，以便一次生成多个噪声级别的图像。

5.随机噪声生成：
epsilons 是与图像张量相同形状的随机噪声。

6.噪声图像生成：
每个时间步 t 计算 a_hat，即累积的 α 值，用于控制噪声和原始图像的加权比例。
根据公式：噪声图像 = sqrt(a_hat) * 原始图像 + sqrt(1 - a_hat) * 噪声。

7.噪声图像后处理：
将噪声图像从 [-1, 1] 映射回 [0, 1]，然后缩放为 [0, 255] 的整数像素值。

8.显示特定噪声级别的图像：
提取生成的噪声图像中的第 7 个（噪声级别 750），转换为 PIL 图像并显示。

"""

# 创建随机数生成器，并设置固定的种子值以确保结果可复现
generator = torch.Generator()
generator.manual_seed(0)

# 实例化 DDPMSampler，用于计算累积 alpha 值和生成噪声样本
ddpm_sampler = DDPMSampler(generator)

# 指定要生成噪声的级别
# 每个值表示扩散过程中的时间步，数值越大，噪声越多
# How many noise levels to generate
noise_levels = [0, 10, 50, 75, 100, 250, 500, 750]

# 加载原始图像并转换为 PyTorch 张量
img = Image.open("../images/cat.jpg")
img_tensor = torch.tensor(np.array(img))
# 将图像像素值从 [0, 255] 归一化到 [-1, 1]
img_tensor = ((img_tensor / 255.0) * 2.0) - 1.0
# Create a batch by repeating the same image many times
# 创建一个 batch，包含多份相同的图像，数量与噪声级别数量相同
batch = img_tensor.repeat(len(noise_levels), 1, 1, 1)

# 转换噪声级别列表为张量（时间步数 t），并将其置于与图像相同的设备上
ts = torch.tensor(noise_levels, dtype=torch.int, device=batch.device)

# 用于存储不同噪声级别的图像
noise_imgs = []

# 为每个时间步生成随机噪声张量，与 batch 形状一致
epsilons = torch.randn(batch.shape, device=batch.device)

# 为每个噪声级别生成对应的噪声图像
# Generate a noisifed version of the image for each noise level
for i in range(len(ts)):
    # 获取当前时间步的 alpha 累积值 a_hat
    a_hat = ddpm_sampler.alphas_cumprod[ts[i]]

    # 根据扩散模型公式生成噪声图像
    # 噪声图像 = sqrt(a_hat) * 原始图像 + sqrt(1 - a_hat) * 噪声
    noise_imgs.append(
        (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
    )

# 将所有噪声图像堆叠到一个张量中，形状为 (噪声级别数, 高, 宽, 通道)
noise_imgs = torch.stack(noise_imgs, dim=0)
# 将噪声图像从 [-1, 1] 重新映射到 [0, 1]，然后转换为像素值 [0, 255]
noise_imgs = (noise_imgs.clamp(-1, 1) + 1) / 2
noise_imgs = (noise_imgs * 255).type(torch.uint8)

# 使用 matplotlib 显示所有噪声级别的图像
plt.figure(figsize=(20, 5))
for idx, (noise_level, img) in enumerate(zip(noise_levels, noise_imgs)):
    plt.subplot(1, len(noise_levels), idx + 1)
    plt.imshow(img.squeeze(0).numpy())
    plt.title(f'噪声级别: {noise_level}')
    plt.axis('off')

plt.tight_layout()
plt.show()

