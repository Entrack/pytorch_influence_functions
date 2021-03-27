import torch
from torch import nn
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='zeros'),
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='zeros'),
)


from torch.utils.data import TensorDataset, DataLoader
traindataset = TensorDataset(torch.rand((8, 1, 7, 7)), torch.ones((8, 1, 7, 7)))
testdataset = TensorDataset(torch.rand((4, 1, 7, 7)), torch.ones((4, 1, 7, 7)))
trainloader = DataLoader(traindataset, batch_size=2)
testloader = DataLoader(testdataset, batch_size=2)


import pytorch_influence_functions as ptif

config = ptif.get_default_config()
config["outdir"] = "./outdir/"
config["gpu"] = -1 # -1 for CPU
config["num_classes"] = None # we don't need classes
config["test_sample_num"] = 2 # let's take first 2 test samples
config["is_pix2pix"] = True

print("config")
print(config)
print("")

influences = ptif.calc_img_wise(config, model, trainloader, testloader)
print("influences")
print(influences)
print("")