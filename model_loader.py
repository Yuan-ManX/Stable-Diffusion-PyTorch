from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter


"""
模型预加载函数 preload_models_from_standard_weights
该函数 preload_models_from_standard_weights 用于从标准权重文件中加载预训练的模型参数，并将其应用到相应的模型实例中。
加载完成后，函数返回包含已加载模型的字典，方便后续使用。

一 导入模块:
clip: 导入 CLIP 模型。
encoder: 导入 VAE 编码器 VAE_Encoder。
decoder: 导入 VAE 解码器 VAE_Decoder。
diffusion: 导入扩散模型 Diffusion。
model_converter: 导入模型转换工具，用于加载标准权重文件。

二 函数参数:
ckpt_path (str): 权重文件的路径。
device (str 或 torch.device): 目标设备，如 'cpu' 或 'cuda:0'。


详细步骤：
1.加载标准权重:
使用 model_converter.load_from_standard_weights 函数从指定路径 ckpt_path 加载标准权重文件，
并将其转换为模型的状态字典 state_dict，加载到指定的设备 device 上。

2.初始化并加载 VAE 编码器:
实例化 VAE_Encoder 并将其移动到指定设备。
使用 load_state_dict 方法，将 state_dict 中键为 'encoder' 的权重加载到编码器中，strict=True 表示严格匹配模型的参数。

3.初始化并加载 VAE 解码器:
实例化 VAE_Decoder 并将其移动到指定设备。
使用 load_state_dict 方法，将 state_dict 中键为 'decoder' 的权重加载到解码器中。

4.初始化并加载扩散模型:
实例化 Diffusion 并将其移动到指定设备。
使用 load_state_dict 方法，将 state_dict 中键为 'diffusion' 的权重加载到扩散模型中。

5.初始化并加载 CLIP 模型:
实例化 CLIP 并将其移动到指定设备。
使用 load_state_dict 方法，将 state_dict 中键为 'clip' 的权重加载到 CLIP 模型中。

6.返回已加载的模型:
返回一个包含已加载的 clip、encoder、decoder 和 diffusion 模型的字典，方便后续调用和使用。

"""

def preload_models_from_standard_weights(ckpt_path, device):
    # 从标准权重文件加载状态字典
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # 初始化 VAE 编码器并移动到指定设备
    encoder = VAE_Encoder().to(device)
    # 加载编码器的状态字典，strict=True 表示严格匹配参数
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    # 初始化 VAE 解码器并移动到指定设备
    decoder = VAE_Decoder().to(device)
    # 加载解码器的状态字典
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # 初始化扩散模型并移动到指定设备
    diffusion = Diffusion().to(device)
    # 加载扩散模型的状态字典
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    # 初始化 CLIP 模型并移动到指定设备
    clip = CLIP().to(device)
    # 加载 CLIP 模型的状态字典
    clip.load_state_dict(state_dict['clip'], strict=True)

    # 返回包含已加载模型的字典
    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }
