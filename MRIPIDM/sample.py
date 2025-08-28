#main for intentional overfitting testing
from modules import *


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.run_name = "DDPM_Uncondtional"
    args.dataset_path = "/root/MRIPIDM/MRIPIDM/test_dataset"
    args.epochs = 1000
    args.batch_size = 3
    #data = torch.tensor(np.load(args.path)) # data.shape: torch.Size([171, 171, 141, 3]), store on CPU and then access each slice index on the GPU
    #data = data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
    #data = pad_to_even(data)

    data = np.zeros((1, 3, 160, 160), dtype=np.float16)


    height = 160
    width = 160
    args.image_size = (height, width)
    args.device = "cuda"
    args.lr = 1e-3
    args.dtype = torch.bfloat16
    train(args, data)


if __name__ == '__main__':
    launch()