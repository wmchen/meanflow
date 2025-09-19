import functools
from typing import Optional

import torch
from einops import rearrange
from .models import DiT


class MeanFlow:
    def __init__(
        self,
        channels: int = 4,
        image_size: int = 32,
        num_classes: int = 1000,
        timestep_equal_ratio: float = 0.75,  # ratio of r=t
        timestep_dist: dict = {"type": "lognorm", "mu": -0.4, "sigma": 1.0},
        cfg_ratio: float = 0.1,
        cfg_w_prime: Optional[float] = 3.0,
        cfg_w: Optional[float] = 3.0,
        cfg_trigger_t_min: float = 0.0,
        cfg_trigger_t_max: float = 1.0,
        p: float = 1.0,
        eps: float = 1e-6
    ):
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.timestep_equal_ratio = timestep_equal_ratio
        self.timestep_dist = timestep_dist
        self.cfg_ratio = cfg_ratio
        self.cfg_w_prime = cfg_w_prime
        self.cfg_w = cfg_w
        self.cfg_k = 1 - cfg_w / cfg_w_prime
        self.cfg_trigger_t_min = cfg_trigger_t_min
        self.cfg_trigger_t_max = cfg_trigger_t_max
        self.p = p
        self.eps = eps

    def sample_timesteps(self, batch_size: int, device):
        if self.timestep_dist["type"] == "lognorm":
            samples = torch.randn((batch_size, 2), device=device) * self.timestep_dist["sigma"] + self.timestep_dist["mu"]
        elif self.timestep_dist["type"] == "uniform":
            samples = torch.randn((batch_size, 2), device=device)

        samples = torch.sigmoid(samples)
        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1])
        num_equals = int(self.timestep_equal_ratio * batch_size)
        indices = torch.randperm(batch_size)[:num_equals]
        r[indices] = t[indices]
        return r*1000, t*1000
    
    def compute_loss(self, model: DiT, x: torch.Tensor, c: torch.Tensor):
        r, t = self.sample_timesteps(x.shape[0], x.device)
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()

        # add noise (forward process)
        e = torch.randn_like(x)
        z = (1 - t_) * x + t_ * e
        v = e - x  # conditional velocity

        # apply CFG
        uncond = torch.ones_like(c) * self.num_classes
        cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
        c = torch.where(cfg_mask, uncond, c)
        if self.cfg_w is not None:
            v_hat = v.detach().clone()
            for i in range(v_hat.shape[0]):
                if self.cfg_trigger_t_min <= t[i].item() <= self.cfg_trigger_t_max:
                    # improved CFG for meanflow
                    z_ = z[i, :, :, :].unsqueeze(0)
                    v_ = v[i, :, :, :].unsqueeze(0)
                    t_i = t[i].unsqueeze(0)
                    uncond_ = uncond[i].unsqueeze(0)
                    cond_ = c[i].unsqueeze(0)
                    with torch.no_grad():
                        u_cfg_cond = model(z_, t_i, t_i, cond_)
                        u_cfg_uncond = model(z_, t_i, t_i, uncond_)
                    v_hat_ = self.cfg_w * v_ + self.cfg_k * u_cfg_cond + (1 - self.cfg_w - self.cfg_k) * u_cfg_uncond
                    v_hat[i, :, :, :] = v_hat_.squeeze(0)
        else:
            v_hat = v
        
        # predict
        model_partial = functools.partial(model, y=c)
        u, dudt = torch.autograd.functional.jvp(
            func=lambda z, r, t: model_partial(x=z, r=r, t=t),
            inputs=(z, r, t),
            v=(v_hat, torch.zeros_like(r), torch.ones_like(t)),
            create_graph=True
        )
        u_tgt = v_hat - (t_ - r_) * dudt
        error = u - u_tgt.detach()
        loss = self.adaptive_l2_loss(error)
        mse_value = (error.detach() ** 2).mean()
        
        return loss, mse_value

    def adaptive_l2_loss(self, error):
        delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
        w = 1.0 / (delta_sq + self.eps).pow(self.p)
        loss = delta_sq  # ||Δ||^2
        return (w.detach() * loss).mean()
