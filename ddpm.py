import torch
import numpy as np


"""
DDPMSampler
DDPMSampler 类负责从训练好的 DDPM 模型中生成样本。
DDPM 是一种生成模型，通过逐步向数据中添加噪声并学习去噪过程来生成高质量的样本。
采样器的主要任务是模拟反向扩散过程，从纯噪声开始，逐步生成接近真实数据的样本。

一 初始化参数: generator, num_training_steps, beta_start, beta_end
二 主要组件: self.betas, self.alphas, self.alphas_cumprod, self.one, self.generator, self.num_train_timesteps, self.timesteps

详细步骤：
一. 初始化参数
初始化 β 和 α：
参数 beta_start 和 beta_end 决定了扩散过程的噪声量，分别表示噪声强度的起始值和结束值。
通过 torch.linspace 按平方根生成均匀分布的 β 序列，再平方得到每个时间步的噪声量 self.betas。
计算 α 值：self.alphas = 1 - self.betas，代表保留原始数据的信息量。
累积 α 值：self.alphas_cumprod = torch.cumprod(self.alphas)，用于表示每一步后保留的总体信息量。
其他参数初始化：
self.one：常量 1.0，供特定情况下使用。
self.timesteps：生成反向的时间步序列，默认为 [num_training_steps-1, ..., 0]。

二. 设置推理时间步数 (set_inference_timesteps 方法)
调整推理时间步数：
通过输入的 num_inference_steps 确定推理步数。
根据比例计算采样时间步（timesteps），使推理的时间步数与训练时间步数对齐。
时间步的反转与类型转换：
时间步按逆序排列，并转换为 PyTorch 张量。

三. 获取前一时间步 (_get_previous_timestep 方法)
计算前一时间步：
当前时间步减去每步的跨度（num_train_timesteps // num_inference_steps）。
用于推导扩散过程中上一时间点的状态。

四. 计算方差 (_get_variance 方法)
累积 α 的获取：
获取当前时间步 t 和前一时间步 prev_t 的累积 α 值。
如果 prev_t 小于 0，则将 alpha_prod_t_prev 设为常量 1。
计算当前 β：
根据公式：current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev。
计算方差：
使用公式 variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t 计算。
为防止数值溢出或无穷，使用 torch.clamp 限制最小值。

五. 设置噪声强度 (set_strength 方法)
根据 strength 调整起始步：
根据噪声强度 strength 确定跳过的噪声级数。
越高的 strength 表示加入的噪声越多，输出更接近纯噪声。
裁剪时间步：
更新 timesteps 以反映调整后的起始点。

六. 单步采样 (step 方法)
获取 α 和 β：
获取当前时间步 t 和前一时间步 prev_t 的累积 α 和 β。
计算预测的原始样本：
根据公式 pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5 计算。
计算系数：
使用公式计算 pred_original_sample 和当前样本 latents 的系数，分别为：
pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
计算预测的上一步样本：
根据公式：pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents。
添加噪声：
根据当前时间步计算噪声，并与预测样本叠加。

七. 添加噪声 (add_noise 方法)
计算噪声权重：
根据当前时间步的累积 α 值，计算原始样本和噪声的系数：
sqrt_alpha_prod：原始样本权重。
sqrt_one_minus_alpha_prod：噪声权重。
生成噪声：
从标准正态分布中采样噪声 noise。
生成带噪声的样本：
根据公式：noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise。

通过上述步骤，DDPMSampler 能够在给定时间步上对噪声数据进行反向扩散，
从而生成与原始数据分布接近的样本。位置嵌入加到 token 嵌入上，得到最终的嵌入向量 x，形状仍为 (Batch_Size, Seq_Len, Dim)。

"""

class DDPMSampler:
    '''
    generator（torch.Generator）：用于生成随机数的 PyTorch 生成器。
    num_training_steps（int）：训练时的总时间步数，默认为 1000。
    beta_start（float）：β 的起始值，默认为 0.00085。
    beta_end（float）：β 的结束值，默认为 0.0120。
    '''

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        # 定义扩散过程中的 β 序列，表示每一步的噪声添加比例。
        # β 序列的生成方法采用线性插值，对起始和结束值进行平方根处理，保证噪声平滑增长。
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        # 定义每一步的 α 值，表示信号的保留比例。
        self.alphas = 1.0 - self.betas
        # 累积 α 的乘积，用于计算去噪过程中的缩放系数。
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 一个常量 1，用于处理边界情况（例如 t=0 时的计算）。
        self.one = torch.tensor(1.0)

        # 随机数生成器，用于采样过程中生成噪声。
        self.generator = generator

        # 定义训练时的总时间步数。
        self.num_train_timesteps = num_training_steps
        # 反向时间步数列表，从最大时间步数递减至 0。
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        '''
        设置推理阶段的时间步数。
        num_inference_steps（int）：推理阶段的时间步数，默认为 50。
        '''
        self.num_inference_steps = num_inference_steps
        # 根据训练时间步数和推理时间步数的比例，确定采样时的时间步数。
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # 根据间隔比率计算推理时间步的索引，并反转顺序。
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        # 更新时间步数列表。
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        '''
        获取当前时间步的前一个时间步。
        timestep（int）：当前时间步索引。
        返回：前一个时间步索引。
        '''
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        '''
        计算给定时间步的方差，基于公式 (6) 和 (7)。
        timestep（int）：当前时间步索引。
        返回：当前时间步的方差。
        '''
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        # 计算方差并将其限制在一个最小值，以避免数值不稳定。
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength=1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        '''
        设置生成样本的强度，决定添加噪声的量。
        strength（float）：噪声强度，范围为 [0, 1]。值越大，生成样本越偏离输入。
        '''
        # start_step is the number of noise levels to skip
        # 根据强度计算起始时间步，并调整时间步索引。
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        '''
        反向扩散的单步计算，基于公式 (7)。
        timestep（int）：当前时间步。
        latents（torch.Tensor）：当前时间步的潜变量。
        model_output（torch.Tensor）：模型预测的噪声。
        返回：前一时间步的潜变量。
        '''
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. compute alphas, betas
        # 计算 alpha 和 beta 的相关值。
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        # 根据模型输出计算原始样本的预测值。    
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        # 根据公式 (7) 计算当前样本和原始样本的系数。
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        # 计算前一时间步的预测值。
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 6. Add noise
        # 添加噪声，模拟随机扩散过程。
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        '''
        向原始样本添加噪声，基于公式 (4)。
        original_samples（torch.FloatTensor）：原始样本张量。
        timesteps（torch.IntTensor）：时间步索引张量。
        返回：添加噪声后的样本。
        '''
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # 计算 alpha 和 (1 - alpha) 的平方根。
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        # 根据公式 (4) 添加噪声。
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

         
