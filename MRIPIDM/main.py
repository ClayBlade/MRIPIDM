#run main
from modules import *

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.run_name = "DDPM_Uncondtional"
    #args.path# = r"/content/MRIPIDM/MRIPIDM/ParametricMaps/slice_24.npy"
    args.path = "/root/MRIPIDM/MRIPIDM/ParametricMaps/slice_24.npy"
    args.epochs = 60
    args.batch_size = 8
    data = torch.tensor(np.load(args.path)) # data.shape: torch.Size([171, 171, 141, 3]), store on CPU and then access each slice index on the GPU
    data = data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
    data = pad_to_even(data)

    #additional_data = torch.tensor(np.load(r"/content/MRIPIDM/MRIPIDM/ParametricMaps/slice_25.npy"))
    #additional_data = additional_data.reshape(additional_data.shape[0], additional_data.shape[3], additional_data.shape[1], additional_data.shape[2])
    #additional_data = additional_data[:160, :, :, :]
    #additional_data = pad_to_even(additional_data)
#
    #print(f"data.shape after padding: {data.shape}") #data.shape after padding: torch.Size([171, 3, 172, 144])
    #print(f"additional_data.shape after padding: {additional_data.shape}") #additional_data.shape after padding: torch.Size([])
#
    #data = torch.cat((data, additional_data), dim=0)
    #print(f"data.shape after concat: {data.shape}") #data.shape after concat: torch.Size([172, 3, 172, 144])

    height = data.shape[2]
    width = data.shape[3]
    args.image_size = (height, width)
    args.device = "cuda"
    args.lr = 1e-3
    args.dtype = torch.bfloat16
    train(args, data)


if __name__ == '__main__':
    launch()


#The Things They Still have to do:
#       Implement more diagnostics: FID, IS scores; loss-time plots; quality vs diversity plots
#       In 16x16 matrix, converge for simple bloch vectors