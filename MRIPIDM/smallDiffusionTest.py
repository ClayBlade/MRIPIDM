#run Model

 #Sanity: identity denoise test passes (loss < small threshold).
 #Reproducible: same seed gives identical sample on tiny model.
 #Canary dataset: compute and log per-epoch PDE residual L2 and boundary errors on 5 canonical cases.
 #Guidnce sweep: run 3 guidance scales and log conditional accuracy / residual.
 #Memory/time: single-sample latency and peak GPU mem profiled.



import os
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from modules import *
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import shutil
shutil.rmtree('runs')

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')




def get_data(args, data):
    # List to store all individual matrices

    data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2]) #data.shape: torch.Size([171, 3, 171, 141])

    print(f"data.shape: {data.shape}") #data.shape: torch.Size([171, 3, 171, 141])

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    return dataloader


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
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device) #since self.img_size is new collection, (1, 2) now (0, 1)
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
        #x = (x.clamp(-1, 1) + 1) / 2
        #x = (x * 255).type(torch.uint8) # Not producing image tho
        return x


def train(args, data):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args, data)
    model = UNet(c_in = 1, c_out = 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device) # (1, 3, 171, 141)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            #print(f" \n t.shape: {t.shape} ")   # t.shape: torch.Size([1])
            #print(f"x_t shape: {x_t.shape} \n") # x_t shape: torch.Size([1, 1, 171, 141, 3])
            predicted_noise = model(x_t, t) #predicted_noise.shape: torch.Size([1, 1, 171, 141, 3])
            #print(f"predicted_noise.shape: {predicted_noise.shape}")
            loss = mse(noise, predicted_noise)
            PSNR = 10 * torch.log10(torch.tensor(1 / loss.item())) #Max is 1 bc max value of M0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("PSNR", PSNR.item(), global_step=epoch * l +i)
            pbar.set_postfix(PSNR=PSNR.item(), MSE=loss.item())

        if (epoch % 10 == 0):
            #print(f"images.shape: {images.shape}")
            #sampled_images = diffusion.sample(model, n=images.shape[0])
            #save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))



def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.run_name = "DDPM_Uncondtional"
    #for colab# args.path = r"/content/MRIPIDM/MRIPIDM/ParametricMaps/slice_0.npy" 
    args.path = "/root/MRIPIDM/MRIPIDM/ParametricMaps/slice_0.npy" 
    args.epochs = 10
    args.batch_size = 1
    data = torch.tensor(np.load(args.path)) # data.shape: torch.Size([171, 171, 141, 3]), store on CPU and then access each slice index on the GPU
    #print(f"data.shape: {data.shape}")
    height = data.shape[1]
    width = data.shape[2]
    args.image_size = (height, width)
    args.device = "cuda"
    args.lr = 3e-4
    args.dtype = torch.float16
    train(args, data)


if __name__ == '__main__':
    launch()


#The Things They Still have to do:
#       Implement more diagnostics: FID, IS scores; loss-time plots; quality vs diversity plots
#       In 16x16 matrix, converge for simple bloch vectors