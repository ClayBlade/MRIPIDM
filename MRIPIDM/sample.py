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

    data = torch.zeros((1, 3, 160, 160), dtype=torch.float32)


    height = 160
    width = 160
    args.image_size = (height, width)
    args.device = "cuda"
    args.lr = 1e-3
    args.dtype = torch.float32
    train(args, data)


if __name__ == '__main__':
    launch()