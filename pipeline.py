import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8



"""
generate 函数是一个用于生成图像的函数，基于扩散模型（Diffusion Model）从文本提示（prompt）生成相应的图像。
它支持条件生成（使用文本提示）和无条件生成（不依赖文本提示），
并允许用户输入初始图像进行图像到图像的转换（image-to-image translation）。
此外，该函数还支持配置控制（CFG, Classifier-Free Guidance）以增强生成图像的质量。

generate 函数通过以下步骤实现图像生成：
1.文本提示处理: 使用 clip 模型将文本提示转换为上下文嵌入。
2.潜在空间初始化: 根据是否有输入图像，初始化潜在空间表示。
3.扩散模型处理: 使用扩散模型和采样器逐步处理潜在空间，生成图像。
4.图像解码: 使用 decoder 模型将潜在空间解码为最终图像。
5.后处理: 对生成的图像进行缩放和维度调整，返回最终的图像数组。

一 主要参数:
prompt (str): 用于生成图像的文本提示。
uncond_prompt (str, 可选): 无条件生成的文本提示，默认为 None。
input_image (PIL.Image.Image, 可选): 输入的初始图像，用于图像到图像的转换，默认为 None。
strength (float): 控制图像到图像转换过程中输入图像的影响程度，范围在0到1之间，默认为0.8。
do_cfg (bool): 是否启用配置控制（CFG），默认为 True。
cfg_scale (float): 配置控制的尺度，默认为7.5。
sampler_name (str): 采样器的名称，默认为 "ddpm"（DDPM 采样器）。
n_inference_steps (int): 推理步骤的数量，默认为50。
models (dict): 包含预加载模型的字典，键包括 'clip', 'encoder', 'decoder', 'diffusion'。
seed (int, 可选): 随机种子，用于结果可复现，默认为 None。
device (str 或 torch.device): 主计算设备，如 'cpu' 或 'cuda:0'。
idle_device (str 或 torch.device, 可选): 备用计算设备，用于将张量移动到空闲设备以节省内存，默认为 None。
tokenizer (transformers.PreTrainedTokenizer, 可选): 用于文本编码的 tokenizer，默认为 None。


详细步骤：
1.参数验证:检查 strength 参数是否在0到1之间，不符合则抛出 ValueError。

2.设备设置:
如果指定了 idle_device，定义一个 to_idle 函数，将张量移动到空闲设备。否则，to_idle 函数不执行任何操作。

3.随机数生成器初始化:初始化随机数生成器 generator，并设置到指定的 device。
如果 seed 为 None，则使用默认种子；否则，使用指定的 seed 手动设置种子以确保结果可复现。

4.模型加载与移动:从 models 字典中获取 clip 模型并移动到指定 device。

5.文本提示处理:
条件生成 (do_cfg=True):
使用 tokenizer 将 prompt 和 uncond_prompt 编码为 token 序列，长度为77（通过填充或截断）。
将 token 序列转换为张量并移动到指定 device。
使用 clip 模型将文本提示转换为上下文嵌入（context embeddings）。
将条件上下文嵌入 cond_context 和无条件上下文嵌入 uncond_context 进行拼接，得到最终的上下文嵌入 context。
无条件生成 (do_cfg=False):仅使用 prompt 进行编码和嵌入处理，得到上下文嵌入 context。

6.模型移动:将 clip 模型移动到 idle_device 以节省内存。

7.采样器初始化:
根据 sampler_name 初始化相应的采样器：
如果是 "ddpm"，则使用 DDPMSampler 并设置推理步骤数量。
否则，抛出 ValueError 提示未知采样器。

8.潜在空间初始化:
定义潜在空间的形状 latents_shape 为 (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)。
图像到图像转换 (input_image 不为 None):
将输入图像调整大小到 (WIDTH, HEIGHT)。
将图像转换为 NumPy 数组，然后转换为张量，并缩放到 [-1, 1] 范围。
添加批次维度并调整通道顺序，得到形状为 (Batch_Size, Channel, Height, Width)。
生成随机噪声 encoder_noise，形状与潜在空间相同。
使用 encoder 模型将输入图像和噪声编码为潜在空间表示 latents。
使用采样器的 add_noise 方法向潜在空间添加噪声。
将 encoder 模型移动到 idle_device 以节省内存。
无条件生成 (input_image 为 None):
使用随机数生成器生成随机潜在空间 latents。

9.扩散模型处理:
将 diffusion 模型移动到指定 device。
使用 tqdm 进度条遍历推理步骤：
获取当前时间步的嵌入 time_embedding 并移动到指定 device。
将潜在空间 latents 作为模型输入：
如果启用 CFG，则将模型输入重复两次（条件和无条件）。
使用 diffusion 模型预测噪声 model_output。
如果启用 CFG，则将模型输出拆分为条件输出 output_cond 和无条件输出 output_uncond，并根据 cfg_scale 进行加权组合。
使用采样器的 step 方法更新潜在空间 latents。

10.模型移动:将 diffusion 模型移动到 idle_device 以节省内存。

11.图像解码:
将 decoder 模型移动到指定 device。
使用 decoder 模型将潜在空间 latents 解码为图像 images，形状为 (Batch_Size, 3, Height, Width)。
将 decoder 模型移动到 idle_device 以节省内存。

12.图像后处理:
将图像缩放到 [0, 255] 范围，并应用 clamp 限制值范围。
调整图像张量的维度顺序为 (Batch_Size, Height, Width, Channel)。
将图像张量转换为 CPU 内存和 uint8 类型，并转换为 NumPy 数组。

13.返回结果:返回生成的图像数组中的第一张图像。

"""

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    


"""
rescale 函数用于将输入张量 x 的数值范围从一个旧的区间 old_range 线性缩放到一个新的区间 new_range。
它支持可选的裁剪（clamping）操作，以确保缩放后的数值不会超出新的范围。

一 参数:
x (Tensor): 需要进行缩放的输入张量。
old_range (tuple): 原始数据范围，表示为 (old_min, old_max)。
new_range (tuple): 目标数据范围，表示为 (new_min, new_max)。
clamp (bool, 可选): 是否对缩放后的数据进行裁剪。如果为 True，则将数值限制在 new_range 内。默认为 False。

"""

def rescale(x, old_range, new_range, clamp=False):
    # 解包原始范围和目标范围
    old_min, old_max = old_range
    new_min, new_max = new_range

    # 将 x 的值减去 old_min，使原始数据的最小值变为0
    x -= old_min

    # 将 x 的值乘以缩放比例，调整数据的范围到新的区间
    x *= (new_max - new_min) / (old_max - old_min)

    # 将 x 的值加上 new_min，将数据的范围平移到新的最小值
    x += new_min

    # 如果 clamp 为 True，则将 x 的数值限制在 new_min 和 new_max 之间
    if clamp:
        x = x.clamp(new_min, new_max)
    
    # 返回缩放后的张量 x
    return x



"""
get_time_embedding 函数用于生成时间步（timestep）的嵌入向量（embedding）。
这种嵌入通常用于时间序列模型或扩散模型中，以将时间信息编码为模型可以理解的数值表示。

get_time_embedding 函数通过以下步骤生成时间步的嵌入向量：
1.频率生成: 生成一个包含160个不同频率的向量 freqs，这些频率用于编码时间步信息。
2.时间步扩展与相乘: 将输入的时间步 timestep 扩展为形状 (1, 1)，并与频率向量 freqs 相乘，得到形状为 (1, 160) 的张量 x。
3.三角函数应用: 对 x 应用余弦和正弦函数，得到两个形状为 (1, 160) 的嵌入向量。
4.拼接嵌入: 将余弦和正弦嵌入在最后一个维度上拼接，得到最终的形状为 (1, 320) 的时间步嵌入向量。
这种嵌入方式在时间序列模型和扩散模型中常用于将时间信息编码为高维空间的数值表示，以便模型更好地捕捉时间依赖关系。

一 参数:
timestep (int 或 float): 当前的时间步，通常是一个标量。


详细步骤：
1.生成频率向量:
创建一个包含160个元素的向量 freqs, 结果 freqs 的形状为 (160,)，表示160个不同的频率。

2.时间步扩展:
将输入的标量 timestep 转换为形状为 (1,) 的张量，并添加一个新的维度，使其形状变为 (1, 1)。
将扩展后的 timestep 与频率向量 freqs 进行逐元素相乘，得到形状为 (1, 160) 的张量 x。

3.生成余弦和正弦嵌入:
对 x 应用余弦函数 torch.cos(x)，得到形状为 (1, 160) 的余弦嵌入。
对 x 应用正弦函数 torch.sin(x)，得到形状为 (1, 160) 的正弦嵌入。

4.拼接嵌入:
将余弦嵌入和正弦嵌入在最后一个维度上进行拼接，得到最终的嵌入向量，形状为 (1, 160 * 2) = (1, 320)。
返回这个形状为 (1, 320) 的嵌入向量。

"""

def get_time_embedding(timestep):
    # Shape: (160,)
    # 生成频率向量，形状为 (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    # 将 timestep 扩展为形状 (1, 1) 并与 freqs 相乘，得到形状 (1, 160) 的张量 x
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    # 对 x 应用余弦和正弦函数，并拼接结果，形状为 (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

