#loading model

from modules import *
import torch

path = r"/root/MRIPIDM/MRIPIDM/ParametricMaps/slice_0.npy"

data = torch.tensor(np.load(path)) # data.shape: torch.Size([171, 171, 141, 3]), store on CPU and then access each slice index on the GPU
data = data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
data = pad_to_even(data)

model = UNet(data.shape[1], data.shape[2]).to(device)

model.load_state_dict(torch.load(r"/root/MRIPIDM/MRIPIDM/models/models/DDPM_Uncondtional/ckpt.pt", map_location=torch.device("cpu")))

model.eval()

height = data.shape[2]
width = data.shape[3]
image_size = (height, width)

diffusion = Diffusion(img_size=image_size, device=device)

sampled_images = diffusion.sample(model, n=5)

mse = nn.MSELoss()

for i, Mhat in enumerate(sampled_images):
  smallest_difference = 1
  for j, M in enumerate(data):
    difference = mse(M, Mhat)
    if (difference < smallest_difference):
        smallest_difference = difference
  print(f"image_{i} smallest_difference: {smallest_difference}")

