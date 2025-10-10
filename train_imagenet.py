import os
import shutil
import argparse
import logging
import math
from datetime import datetime, timedelta
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from PIL import Image
from mlcbase import create

from mean_flow import MeanFlow, DiT_models

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--tracker_project_name", type=str, default="MeanFlow")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--vae", type=str, required=True)

    # data settings
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # meanflow settings
    parser.add_argument("--timestep_equal_ratio", type=float, default=0.75)
    parser.add_argument("--timestep_dist_type", type=str, choices=["lognorm", "uniform"], default="lognorm")
    parser.add_argument("--timestep_dist_mu", type=float, default=-0.4)
    parser.add_argument("--timestep_dist_sigma", type=float, default=1.0)
    parser.add_argument("--cfg", action="store_true")
    parser.add_argument("--cfg_w", type=float, default=3.0)
    parser.add_argument("--cfg_k", type=float, default=0.0)
    parser.add_argument("--cfg_trigger_t_min", type=float, default=0.0)
    parser.add_argument("--cfg_trigger_t_max", type=float, default=1.0)
    parser.add_argument("--cfg_cond_dropout", type=float, default=0.1)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)

    # training settings
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")

    # optimizer settings
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")

    # checkpoint settings
    parser.add_argument("--checkpointing_steps", type=int, default=10000)
    parser.add_argument("--resume_from_checkpoint", type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."))

    # validation settings
    parser.add_argument("--val_inference_steps", type=int, nargs="+", default=None)
    parser.add_argument("--val_class_labels", type=int, nargs="+", default=None)
    parser.add_argument("--val_images_each_row", type=int, default=4)

    # others
    parser.add_argument("--remarks", type=str, default=None)

    args = parser.parse_args()

    return args


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def format_time(time: timedelta, timeformat: str ="%d days %H:%M:%S"):
    if "%d" in timeformat:
        has_day = True
    else:
        has_day = False

    if "%H" in timeformat:
        has_hour = True
    else:
        has_hour = False

    if "%M" in timeformat:
        has_minute = True
    else:
        has_minute = False
    
    if "%S" in timeformat:
        has_second = True
    else:
        has_second = False

    if "%f" in timeformat:
        has_microsecond = True
    else:
        has_microsecond = False

    delta_days = time.days
    delta_seconds = time.seconds
    microseconds = time.microseconds
    hours = delta_seconds // 3600
    minute = (delta_seconds % 3600) // 60
    seconds = delta_seconds % 60
    show_day = 0
    show_hour = 0
    show_minute = 0
    show_second = 0
    show_microsecond = 0

    # if %d not in timeformat, unsqueeze days to hours
    if has_day:
        show_day = delta_days
    else:
        hours += delta_days * 24

    # if %H not in timeformat, unsqueeze hours to minutes
    if has_hour:
        show_hour = hours
    else:
        minute += hours * 60

    # if %M not in timeformat, unsqueeze minutes to seconds
    if has_minute:
        show_minute = minute
    else:
        seconds += minute * 60

    # if %S not in timeformat, unsqueeze seconds to microseconds
    if has_second:
        show_second = seconds
    else:
        microseconds += seconds * 10**6

    # ignore microsecond when %f not exists in timeformat
    if has_microsecond:
        show_microsecond = microseconds

    show_day = str(show_day)
    show_hour = f"{show_hour:02d}"
    show_minute = f"{show_minute:02d}"
    show_second = f"{show_second:02d}"
    show_microsecond = str(show_microsecond)

    return show_day, show_hour, show_minute, show_second, show_microsecond


def format_show_time(days, hours, minutes, seconds, microseconds, timeformat="%d days %H:%M:%S"):
    show_text = ""
    format_flag = False
    format_symbol = ""
    for ch in timeformat:
        if ch == "%":
            format_flag = True
            format_symbol += "%"
            continue

        if format_flag:
            format_symbol += ch
            if format_symbol == "%d":
                show_text += (days)
            elif format_symbol == "%H":
                show_text += hours
            elif format_symbol == "%M":
                show_text += minutes
            elif format_symbol == "%S":
                show_text += seconds
            elif format_symbol == "%f":
                show_text += microseconds
            else:
                show_text += format_symbol
            format_symbol = ""
            format_flag = False
        else:
            show_text += ch
            
    return show_text


def main():
    args = parse_args()

    # setting accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "log"))
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir):
            create(os.path.join(args.output_dir, "log"), "dir")
    
    set_seed(args.seed)

    # load dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_dataset = ImageFolder(args.data_path, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_path})")

    # load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        is_cond=True
    ).to(accelerator.device)
    model.train()
    if args.use_ema:
        ema = deepcopy(model).requires_grad_(False)
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()
        if args.offload_ema:
            ema.pin_memory()
        else:
            ema.to(accelerator.device)
        
    vae = AutoencoderKL.from_pretrained(f"{args.vae}").to(accelerator.device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
    num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # meanflow
    meanflow = MeanFlow(
        channels=4,
        image_size=args.image_size,
        num_classes=1000,
        timestep_equal_ratio=args.timestep_equal_ratio,
        timestep_dist_type=args.timestep_dist_type,
        timestep_dist_mu=args.timestep_dist_mu,
        timestep_dist_sigma=args.timestep_dist_sigma,
        cfg=args.cfg,
        cfg_w=args.cfg_w,
        cfg_k=args.cfg_k,
        cfg_trigger_t_min=args.cfg_trigger_t_min,
        cfg_trigger_t_max=args.cfg_trigger_t_max,
        cfg_cond_dropout=args.cfg_cond_dropout,
        p=args.p,
        eps=args.eps
    )

    # tracker
    if accelerator.is_main_process:
        args_dict = dict(vars(args))
        for k, v in args_dict.items():
            if isinstance(v, list):
                args_dict[k] = ", ".join([str(i) for i in v])
        accelerator.init_trackers(args.tracker_project_name, args_dict)

    # train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint") and d.endswith(".pth")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[0].split("-")[1]))
            path = checkpoints[-1] if len(checkpoints) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            ckpt = torch.load(os.path.join(args.output_dir, path), map_location="cpu")
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt['ema'])
            optimizer.load_state_dict(ckpt['opt'])
            global_step = ckpt['steps']

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # prepare
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    
    # start training loop
    start_time = datetime.now()
    last_update_time = deepcopy(start_time)
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        train_mse_loss = 0.0
        for step, (x, y) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                x = x.to(accelerator.device, non_blocking=True)
                y = y.to(accelerator.device, non_blocking=True)
                with torch.no_grad():
                # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                
                loss, mse_loss = meanflow.compute_loss(model, x, y)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_mse_loss = accelerator.gather(mse_loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_mse_loss += avg_mse_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    if args.offload_ema:
                        ema.to(device="cuda", non_blocking=True)
                    update_ema(ema, model, decay=args.ema_decay)
                    if args.offload_ema:
                        ema.to(device="cpu", non_blocking=True)

                cur_time = datetime.now()
                elapsed = cur_time - start_time
                days, hours, minutes, seconds, microseconds = format_time(elapsed)
                elapsed_text = format_show_time(days, hours, minutes, seconds, microseconds)
                elapsed_unit_time = cur_time - last_update_time
                _, _, _, seconds, microseconds = format_time(elapsed_unit_time, "%S%f")
                seconds = int(seconds) + int(microseconds) / 10**6
                if seconds == 0:
                    left_time_text = "unknown"
                else:
                    velocity = 1.0 / seconds
                    left_time = timedelta(seconds=int((max_train_steps-global_step-1) / velocity))
                    days, hours, minutes, seconds, microseconds = format_time(left_time)
                    left_time_text = format_show_time(days, hours, minutes, seconds, microseconds)
                last_update_time = deepcopy(cur_time)

                logger.info(
                    f"Global Step: {global_step+1}/{max_train_steps}, Epoch: {epoch}, Loss: {train_loss:.4f}, "
                    f"MSE Loss: {train_mse_loss:.4f} (Elapsed: {elapsed_text}, Left: {left_time_text}, {velocity:.2f} step/s)"
                )
                global_step += 1
                accelerator.log(
                    {"train_loss": train_loss, 
                     "train_mse_loss": train_mse_loss, 
                     "lr": lr_scheduler.get_last_lr()[0]}, 
                    step=global_step
                )
                train_loss = 0.0
                train_mse_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint") and d.endswith(".pth")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[0].split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    os.remove(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pth")
                        ckpt = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args,
                            "steps": global_step,
                        }
                        torch.save(ckpt, save_path)
                        logger.info(f"Saved state to {save_path}")
                
            if global_step >= max_train_steps:
                break
        
        # validation
        if args.val_class_labels is not None and args.val_inference_steps is not None:
            if accelerator.is_main_process:
                logger.info("Running validation... ")
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        break

                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    val_model = ema
                else:
                    val_model = accelerator.unwrap_model(model)
                    val_model.eval()

                class_labels = [int(i) for i in args.val_class_labels]
                for num_steps in args.val_inference_steps:
                    logger.info(f"Inference with {num_steps} steps...")
                    images = meanflow.sample(
                        model=val_model, 
                        vae=vae, 
                        labels=class_labels,
                        num_sample_steps=num_steps,
                        num_images_per_label=1,
                        device=accelerator.device,
                        output_type="pt"
                    )
                    images = make_grid(images, nrow=args.val_images_each_row).unsqueeze(0)
                    images = images.mul(0.5).add_(0.5).clamp_(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    tracker.log_images({f"val_{num_steps}-step": images}, step=global_step, dataformats="NHWC")

                if not args.use_ema:
                    del val_model
    
    accelerator.wait_for_everyone()

    # final save
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        ckpt = {"model": model.state_dict(), "ema": ema.state_dict()}
        torch.save(
            ckpt, 
            os.path.join(
                args.output_dir, 
                f"MF_{args.model.replace('/', '-')}_res{args.image_size}_e{args.num_train_epochs}.pth"
            )
        )
        logger.info("Training completed!")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
