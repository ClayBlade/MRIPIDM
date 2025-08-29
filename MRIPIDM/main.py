#run main
def launch(data, model_run_type):
    import argparse
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.run_name = "DDPM_Uncondtional"

    args.epochs = 50
    args.batch_size = 2
    
    height = data.shape[2]
    width = data.shape[3]
    args.image_size = (height, width)
    args.device = "cuda"
    args.lr = 5e-4
    args.dtype = torch.bfloat16
    train(args, data, model_run_type)


if __name__ == '__main__':
    model_run_type = "main"
    model_run_location = "VM"
    if model_run_location == "colab":
      path = r"/content/MRIPIDM/MRIPIDM/ParametricMaps/slice_24.npy"
    else:
      from modules import *
      path = r"/root/MRIPIDM/MRIPIDM/ParametricMaps/slice_24.npy"
    data = torch.tensor(np.load(path)) # data.shape: torch.Size([171, 171, 141, 3]), store on CPU and then access each slice index on the GPU
    data = data.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
    data = pad_to_even(data)
    launch(data, model_run_type)