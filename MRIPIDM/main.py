#run main
from modules import *

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.run_name = "DDPM_Uncondtional"
    #for colab# args.path = r"/content/MRIPIDM/MRIPIDM/ParametricMaps/slice_0.npy"
    args.path = "/root/MRIPIDM/MRIPIDM/ParametricMaps/slice_0.npy"
    args.epochs = 30
    args.batch_size = 1
    data = torch.tensor(np.load(args.path)) # data.shape: torch.Size([171, 171, 141, 3]), store on CPU and then access each slice index on the GPU
    data = data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
    data = pad_to_even(data)
    print(f"data.shape after padding: {data.shape}") #data.shape after padding: torch.Size([171, 3, 172, 144])
    height = data.shape[2]
    width = data.shape[3]
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