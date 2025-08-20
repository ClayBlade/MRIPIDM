#loading model

from modules import *
import torch

NUM_SAMPLES = 5

path = r"/root/MRIPIDM/MRIPIDM/ParametricMaps/slice_24.npy"

data = torch.tensor(np.load(path)) # data.shape: torch.Size([171, 171, 141, 3]), store on CPU and then access each slice index on the GPU
data = data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
data = pad_to_even(data)

model = UNet(data.shape[1], data.shape[2]).to(device)

model.load_state_dict(torch.load(r"/root/MRIPIDM/MRIPIDM/models/content/models/DDPM_Uncondtional/ckpt.pt", map_location=torch.device("cpu")))

model.eval()

height = data.shape[2]
width = data.shape[3]
image_size = (height, width)

diffusion = Diffusion(img_size=image_size, device=device)

sampled_images = diffusion.sample(model, n = NUM_SAMPLES)

mse = nn.MSELoss()
