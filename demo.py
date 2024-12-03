import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch


"""
该脚本使用预训练的扩散模型（Diffusion Model）从文本提示生成图像，
或基于输入图像进行图像到图像的转换（image-to-image translation）。
用户可以指定文本提示、初始图像、生成参数等，生成高质量的图像。

该脚本通过以下步骤实现图像生成：
1.设备选择: 根据系统配置选择合适的计算设备（CPU, CUDA, MPS）。
2.模型加载: 使用预训练的模型权重初始化模型。
3.文本提示处理: 使用 CLIPTokenizer 将文本提示转换为模型可理解的 token。
4.图像生成:
文本到图像: 根据文本提示生成图像。
图像到图像: 根据输入图像和文本提示进行图像转换。
5.采样器设置: 使用指定的采样器和推理步骤数量进行图像生成。
6.结果展示: 将生成的图像转换为 PIL 图像对象并显示。
这种脚本结构适用于各种图像生成任务，包括文本到图像和图像到图像的转换，并且可以通过调整参数来控制生成图像的质量和风格。

详细步骤：
1.导入必要的模块:
model_loader: 用于加载模型。
pipeline: 包含生成图像的管道。
PIL.Image: 用于处理图像数据。
pathlib.Path: 用于处理文件路径。
transformers.CLIPTokenizer: 用于文本编码。
torch: PyTorch库，用于张量计算和模型操作。

2.设备设置:
默认设备为 "cpu"。
如果允许使用 CUDA 且 CUDA 可用，则将设备设置为 "cuda"。
如果允许使用 MPS（Metal Performance Shaders）且 MPS 可用，则将设备设置为 "mps"。
打印当前使用的设备。

3.加载 Tokenizer:
使用指定的 vocab.json 和 merges.txt 文件初始化 CLIPTokenizer，用于将文本提示转换为模型可理解的 token。

4.加载预训练模型:
指定模型文件路径 model_file（例如 v1-5-pruned-emaonly.ckpt）。
使用 model_loader.preload_models_from_standard_weights 函数加载模型权重，并将其存储在 models 字典中。

5.文本到图像生成:
文本提示:
prompt: 用于生成图像的文本描述，例如 "一只戴着墨镜、戴着舒适帽子的狗，看着镜头，高清，8K分辨率"。
uncond_prompt: 无条件生成的文本提示（也称为负提示），此处为空字符串。
配置控制（CFG）:
do_cfg: 是否启用配置控制，启用后可以提高生成图像的质量。
cfg_scale: 配置控制的尺度，范围为1到14，数值越高，生成的图像越符合文本提示，但可能降低多样性。

6.图像到图像转换:
输入图像:
input_image: 输入的初始图像路径（例如 dog.jpg）。
如果注释掉 input_image = Image.open(image_path)，则禁用图像到图像转换，仅进行文本到图像生成。
强度（strength）:
控制输入图像对最终生成图像的影响程度，范围为0到1。
数值越高，输入图像的影响越小，生成图像越接近随机噪声；数值越低，输入图像的影响越大，生成图像越接近输入图像。

7.采样器设置:
sampler: 采样器的名称，此处使用 "ddpm"（Denoising Diffusion Probabilistic Model）。
num_inference_steps: 推理步骤的数量，默认为50，数值越高，生成图像的质量越高，但计算时间越长。
seed: 随机种子，用于结果可复现，设置为42。

8.生成图像:
调用 pipeline.generate 函数，传入上述所有参数，生成图像。
返回的 output_image 是一个 NumPy 数组，形状为 (Height, Width, Channel)。

9.显示输出图像:
使用 Image.fromarray 将 NumPy 数组转换为 PIL 图像对象，并显示。

"""

# 设置设备，默认为 "cpu"
DEVICE = "cpu"

# 是否允许使用 CUDA 和 MPS
ALLOW_CUDA = False
ALLOW_MPS = False

# 如果允许使用 CUDA 且 CUDA 可用，则使用 "cuda" 设备
if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
# 如果允许使用 MPS 且 MPS 可用，则使用 "mps" 设备
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# 初始化 CLIP Tokenizer
tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
# 指定模型文件路径
model_file = "../data/v1-5-pruned-emaonly.ckpt"
# 加载预训练模型
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE 文生图

# 文本提示
prompt = "A cat with sunglasses."
# 无条件生成的文本提示（负提示）
uncond_prompt = ""
# 是否启用配置控制
do_cfg = True
# 配置控制的尺度
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE 图生图

input_image = None
# Comment to disable image to image
# 注释掉以下行以禁用图像到图像转换
image_path = "../images/cat.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
# 强度值越高，添加的噪声越多，生成图像越远离输入图像
# 强度值越低，添加的噪声越少，生成图像越接近输入图像
strength = 0.9

## SAMPLER 采样器

sampler = "ddpm"
num_inference_steps = 50
seed = 42

# 生成图像
output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
# 将输出图像和输入图像合并为一个图像。
Image.fromarray(output_image)
