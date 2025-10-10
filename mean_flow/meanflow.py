import functools
from typing import Union, List

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from einops import rearrange
from PIL import Image
from .models import DiT


class MeanFlow:
    def __init__(
        self,
        channels: int = 4,
        image_size: int = 256,
        num_classes: int = 1000,
        timestep_equal_ratio: float = 0.75,  # ratio of r=t
        timestep_dist_type: str = "lognorm",
        timestep_dist_mu: float = -0.4,
        timestep_dist_sigma: float = 1.0,
        cfg: bool = True,
        cfg_w: float = 3.0,
        cfg_k: float = 0.0,
        cfg_trigger_t_min: float = 0.0,
        cfg_trigger_t_max: float = 1.0,
        cfg_cond_dropout: float = 0.1,
        p: float = 1.0,
        eps: float = 1e-6
    ):
        if cfg:
            assert cfg_w >= 1.0 and 0.0 <= cfg_k < 1.0

        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.timestep_equal_ratio = timestep_equal_ratio
        self.timestep_dist_type = timestep_dist_type
        self.timestep_dist_mu = timestep_dist_mu
        self.timestep_dist_sigma = timestep_dist_sigma
        self.cfg = cfg
        self.cfg_w = cfg_w
        self.cfg_k = cfg_k
        self.cfg_trigger_t_min = cfg_trigger_t_min
        self.cfg_trigger_t_max = cfg_trigger_t_max
        self.cfg_cond_dropout = cfg_cond_dropout
        self.p = p
        self.eps = eps

    def sample_timesteps(self, batch_size: int, device):
        if self.timestep_dist_type == "lognorm":
            samples = torch.randn((batch_size, 2), device=device) * self.timestep_dist_sigma + self.timestep_dist_mu
        elif self.timestep_dist_type == "uniform":
            samples = torch.randn((batch_size, 2), device=device)

        samples = torch.sigmoid(samples)
        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1])
        num_equals = int(self.timestep_equal_ratio * batch_size)
        indices = torch.randperm(batch_size)[:num_equals]
        r[indices] = t[indices]
        return r, t
    
    def compute_loss(self, model: DiT, x: torch.Tensor, c: torch.Tensor):
        r, t = self.sample_timesteps(x.shape[0], x.device)
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()

        # add noise (forward process)
        e = torch.randn_like(x)
        z = (1 - t_) * x + t_ * e
        v = e - x  # conditional velocity

        # apply CFG
        if self.cfg:
            uncond = torch.ones_like(c) * self.num_classes
            w = torch.where((self.cfg_trigger_t_min <= t) & (t <= self.cfg_trigger_t_max), self.cfg_w, 1.0)
            w = w.view(-1, 1, 1, 1)
            if self.cfg_k == 0:
                # normal CFG
                u_cfg_uncond = model(z, t, t, uncond)
                v_hat = w * v + (1.0 - w) * u_cfg_uncond
            else:
                # improved CFG in Appendix
                k = torch.where((self.cfg_trigger_t_min <= t) & (t <= self.cfg_trigger_t_max), self.cfg_k, 0.0)
                k = k.view(-1, 1, 1, 1)
                u_cfg_cond = model(z, t, t, c)
                u_cfg_uncond = model(z, t, t, uncond)
                v_hat = w * v + k * u_cfg_cond + (1.0 - w - k) * u_cfg_uncond
            
            # apply conditional dropout
            mask = torch.rand_like(c.float()) < self.cfg_cond_dropout
            c = torch.where(mask, uncond, c)
            v_hat = torch.where(mask.view(-1, 1, 1, 1), v, v_hat)
        else:
            v_hat = v
        
        # predict
        model_partial = functools.partial(model, y=c)
        u, dudt = torch.func.jvp(
            func=lambda z, r, t: model_partial(x=z, r=r, t=t),
            primals=(z, r, t),
            tangents=(v_hat, torch.zeros_like(r), torch.ones_like(t)),
        )
        u_tgt = v_hat - (t_ - r_) * dudt
        error = u - u_tgt.detach()
        loss = self.adaptive_l2_loss(error)
        mse_value = (error ** 2).mean()
        
        return loss, mse_value

    @torch.no_grad()
    def sample(
        self, 
        model: DiT, 
        vae: AutoencoderKL,
        labels: Union[int, List[int]],
        num_sample_steps: int = 1,
        num_images_per_label: int = 1, 
        vae_scaling_factor: float = 0.18215,
        device: str = "cuda",
        output_type: str = "pil"
    ) -> List[Image.Image]:
        if isinstance(labels, int):
            labels = [labels]
        
        if output_type == "pil":
            resuls = []
        elif output_type == "np" or output_type == "pt":
            resuls = None

        vae_resolution_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        latent_size = self.image_size // vae_resolution_scale_factor
        for i, label in enumerate(labels):
            print(f"Sampling label {label} ({i+1}/{len(labels)})...")

            z = torch.randn(num_images_per_label, self.channels, latent_size, latent_size, device=device)
            y = torch.tensor([label] * num_images_per_label, device=device)
            timesteps = torch.linspace(1.0, 0.0, num_sample_steps+1, dtype=torch.float32)

            for j in range(num_sample_steps):
                t = torch.full((num_images_per_label, ), timesteps[j], device=device)
                r = torch.full((num_images_per_label, ), timesteps[j+1], device=device)
                u = model(z, r, t, y)
                t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
                r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
                z = z - (t_ - r_) * u
            
            images = vae.decode(z / vae_scaling_factor).sample
            if output_type == "pil" or output_type == "np":
                images = images.mul(0.5).add_(0.5).clamp_(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                if output_type == "pil":
                    for j in range(num_images_per_label):
                        image = Image.fromarray(images[j])
                        resuls.append(image)
                else:
                    if resuls is None:
                        resuls = images
                    else:
                        resuls = np.concatenate([resuls, images], axis=0)  
            elif output_type == "pt":
                if resuls is None:
                    resuls = images
                else:
                    resuls = torch.cat([resuls, images], dim=0)
        
        return resuls

    def adaptive_l2_loss(self, error):
        delta_sq = torch.sum((error**2).reshape(error.shape[0], -1), dim=-1)
        w = 1.0 / (delta_sq.detach() + self.eps).pow(self.p)
        loss = w * delta_sq
        return loss.mean()
