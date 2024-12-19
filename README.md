# Stable Diffusion PyTorch

<p align="center">
  <img src="StableDiffusion.png" alt="StableDiffusion" style="display:block; margin:auto; width:620px;" />
</p>

PyTorch implementation of Stable Diffusion

[Stable Diffusion](https://arxiv.org/abs/2112.10752): High-Resolution Image Synthesis with Latent Diffusion Models.

Stable Diffusion is a latent text-to-image diffusion model. Thanks to a generous compute donation from Stability AI and support from LAION, they were able to train a Latent Diffusion Model on 512x512 images from a subset of the LAION-5B database. Similar to Google's Imagen, this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM. 
