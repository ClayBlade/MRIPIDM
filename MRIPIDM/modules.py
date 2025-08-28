#Run Module Util

import torch
import torch.nn as nn
import torch.nn.functional as F







class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.H = size[0]
        self.W = size[1]
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return x.view(B, H, W, C).permute(0, 3, 1, 2)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            #print(f"x.shape in DoubleConv from Up: {x.shape}")
            #print(f"double_conv.shape: {self.double_conv(x).shape}")
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.out_channels = out_channels

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        #print( f" x.shape to be concated: {x.shape}")
        #print(f"skip_x.shape: {skip_x.shape}")
        x = torch.cat([skip_x, x], dim=1)
        #x = nn.Conv2d(in_channels=x.shape[1], out_channels=self.out_channels, kernel_size=1, device = "cuda")(x) # 1x1 conv to match channels
        #print(f"x.shape after 1x1 conv: {x.shape}")
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up_no_scale(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.out_channels = out_channels

        self.up = nn.Upsample(scale_factor=1, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        #print( f" x.shape to be concated: {x.shape}")
        #print(f"skip_x.shape: {skip_x.shape}")
        x = torch.cat([skip_x, x], dim=1)
        #Should this be commented out# x = nn.Conv2d(in_channels=x.shape[1], out_channels=self.out_channels, kernel_size=1, device = "cuda")(x) # 1x1 conv to match channels
        #print(f"x.shape after 1x1 conv: {x.shape}")
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
  def __init__(self, H, W, c_in=3, c_out=3, time_dim=256, device="cuda"):
    super().__init__()
    self.device = device
    self.time_dim = time_dim

    self.device = device
    self.time_dim = time_dim
    self.inc = DoubleConv(c_in, 64)
    self.down1 = Down(64, 128)
    self.sa1 = SelfAttention(128, (H//2, W//2))
    self.down2 = Down(128, 256)
    self.sa2 = SelfAttention(256, (H//4, W//4))
    #self.down3 = Down(256, 256)
    #self.sa3 = SelfAttention(256, (H//8, W//8))  # Adjusted for 3D input

    self.bot1 = DoubleConv(256, 512)
    self.bot2 = DoubleConv(512, 256)
    self.bot3 = DoubleConv(256, 128)

    #self.up1 = Up(512, 128)
    #self.sa4 = SelfAttention(128, (H/4, W/4))
    self.up2 = Up(256, 64)
    self.sa5 = SelfAttention(64, (H/2, W/2))
    self.up3 = Up_no_scale(128, 32)
    #self.sa6 = SelfAttention(32, (H, W))
    self.outc = nn.Conv2d(32, c_out, kernel_size=1)


  def pos_encoding(self, t, channels):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc


  def forward(self, x, t):
    #print(f"x.shape[2]: {x.shape[2]}") #171
    #print(f"x.shape[3]: {x.shape[3]}") #141
    #print(f"x.shape[4]: {x.shape[4]}") #3
    #print(f"x.shape[1]: {x.shape[1]}") #1
    #print(f"device: {self.device}")
    #x = pad_to_even(x)
    t = t.unsqueeze(-1).type(torch.float) #maybe another squeeze for 3D?
    t = self.pos_encoding(t, self.time_dim)
    #print(f"t.shape: {t.shape}")
    #print(f"input x.shape: {x.shape}") #input x.shape: torch.Size([1, 3, 172, 144])

    x1 = self.inc(x)
    #print(f"\n inc: {x1.shape} \n") # inc: torch.Size([1, 64, 172, 144])
    x2 = self.down1(x1, t)
    #print(f"\n down1: {x2.shape} \n") # down1: torch.Size([1, 128, 86, 72])
    x2 = self.sa1(x2)
    #print(f"\n sa1: {x2.shape} \n") # sa1: torch.Size([1, 128, 86, 72])
    x3 = self.down2(x2, t)
    #print(f"\n down2: {x3.shape} \n") #  down2: torch.Size([1, 256, 43, 36])
    x3 = self.sa2(x3)
    #print(f"\n sa2: {x3.shape} \n") # sa2: torch.Size([1, 256, 43, 36])
    #x4 = self.down3(x3, t)
    #print(f"\n down3: {x2.shape} \n")
    #x4 = self.sa3(x4)
    #print(f"\n sa3: {x2.shape} \n")

    x4 = self.bot1(x3)
    x4 = self.bot2(x4)
    x4 = self.bot3(x4)

    #print(f"\n bot3: {x4.shape} \n") # (1, 128, 43, 36)

    #x = self.up1(x4, x3, t)
    #print(f"\n up1: {x.shape} \n")
    #x = self.sa4(x)
    #print(f"\n sa4: {x.shape} \n")
    x = self.up2(x4, x2, t)
    #print(f"\n up2: {x.shape} \n") # up2: torch.Size([1, 64, 86, 72])
    x = self.sa5(x)
    #print(f"\n sa5: {x.shape} \n") # sa5: torch.Size([1, 64, 86, 72])
    upsample = nn.Upsample(scale_factor=2, mode="nearest")
    x = upsample(x)
    #print(f"\n upsample: {x.shape} \n")
    x = self.up3(x, x1, t)
    #print(f"\n up3: {x.shape} \n") # up3: torch.Size([1, 32, 172, 144])
    #x = self.sa6(x)
    #print(f"\n sa6: {x.shape} \n")

    output = self.outc(x)
    return output






#run main utils




import os
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

import math
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import shutil
os.makedirs("runs", exist_ok=True)
shutil.rmtree('runs')

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def pad_to_even(x):
    # Pad each spatial dimension if it's odd
    _, _, H, W = x.shape
    pad_h = (0, 1) if H % 2 != 0 else (0, 0)
    pad_w = (0, 3) if W % 2 != 0 else (0, 0)
    # F.pad pads last dim first: (W, H, D)
    return F.pad(x, pad_w + pad_h, mode='constant', value=0)

def get_data(args, data):
    # List to store all individual matrices

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    return dataloader



def cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-4):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr / base_lr + (1 - min_lr / base_lr) * cosine
    return LambdaLR(optimizer, lr_lambda)

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        '''Fully noises from clean image in one function'''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] #scaler for x based on t
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None] #scaler for noise
        Ɛ = torch.randn_like(x) # noise
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ # returns noised image, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device) #self.img_size is (H, W)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        return x



def train(args, data):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args, data)
    #dataloader = get_data(args)
    model = UNet(data.shape[2], data.shape[3]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = cosine_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs, base_lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    total_norm = 0

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar): #unpack (images, _) when dealing with images
            images = images.to(device) # (8, 3, 172, 144)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)

            #print(f"images.shape: {images.shape}")
            #print(f"x_t.shape: {x_t.shape}")
            #print(f"noise.shape: {noise.shape}")
            #print(f"predicted_noise.shape: {predicted_noise.shape}")

            loss = mse(noise, predicted_noise)
            PSNR = 10 * torch.log10(torch.tensor(1 / loss.item())) #Max is 1 bc max value of M0

            if i > 20: #Validation set
              model.eval()

              logger.add_scalar("MSE validate", loss.item(), global_step=epoch * l + i)
              logger.add_scalar("PSNR validate", PSNR.item(), global_step=epoch * l + i)
              pbar.set_postfix(PSNR=PSNR.item(), MSE=loss.item())

            else: #Training set
              model.train()


              optimizer.zero_grad()
              loss.backward()
              optimizer.step()        

              grads = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]

              logger.add_scalar("debug/grad_mean", float(np.mean(grads)), global_step=epoch * l + i)
              logger.add_scalar("debug/grad_median", float(np.median(grads)), global_step=epoch * l + i)
              logger.add_scalar("debug/grad_min", float(np.min(grads)), global_step=epoch * l + i)
              logger.add_scalar("debug/grad_max", float(np.max(grads)), global_step=epoch * l + i)

              zero_count = sum(1 for p in model.parameters() if p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)))
              logger.add_scalar("debug/zero_grad_param_count", zero_count, global_step=epoch * l + i)

              logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
              logger.add_scalar("PSNR", PSNR.item(), global_step=epoch * l + i)
              pbar.set_postfix(PSNR=PSNR.item(), MSE=loss.item())
        
        scheduler.step()
        print(f"  Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")


        for name, param in model.named_parameters():
          if param.grad is not None:
            logger.add_histogram(f"grads/{name}_distr", param.grad, global_step = epoch)
            param_norm = param.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
          logger.add_scalar(f"grads/{name}_grad_norm", total_norm, global_step = epoch )



        if (epoch % 10 == 9):
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

        