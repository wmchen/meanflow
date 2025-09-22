import argparse

import torch
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from diffusers.models import AutoencoderKL

from mean_flow import DiT_models


@torch.no_grad()
def main(args):
    # Setup PyTorch:
    if args.seed:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model_name](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if "ema" in state_dict.keys():
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [int(i) for i in args.class_labels]

    # Create sampling noise:
    z = torch.randn(len(class_labels), 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # denoise
    timesteps = torch.linspace(1.0, 0.0, args.num_steps+1, dtype=torch.float32)
    with tqdm(total=args.num_steps) as pbar:
        for i in range(args.num_steps):
            t = torch.full((len(class_labels), ), timesteps[i], device=device)
            r = torch.full((len(class_labels), ), timesteps[i+1], device=device)
            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
            u = model(z, r, t, y)
            z = z - (t_ - r_) * u
            pbar.update(1)

    # decode to image
    images = vae.decode(z / 0.18215).sample
    images = make_grid(images, nrow=args.num_images_each_row, normalize=True, value_range=(-1, 1))
    images = images.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    images = Image.fromarray(images)
    images.save(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # models
    parser.add_argument("--model_name", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae", type=str, required=True)

    parser.add_argument("--class_labels", nargs="+", required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=2.5)

    # output
    parser.add_argument("--num_images_each_row", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="result.png")
    args = parser.parse_args()
    main(args)
