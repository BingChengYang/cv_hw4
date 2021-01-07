import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transform
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse
import numpy as np
from PIL import Image
from os import path
# my own library
import load_data_1
import net

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model.pkl")
args = parser.parse_args()

mini_batch_size = 1

transform = transform.Compose([
    transform.ToTensor(),
])

network = torch.load('./model/'+args.model)
print(network)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
network.to(device)

test_set = load_data_1.Load_testdata(transform=transform)
test_load = Data.DataLoader(dataset=test_set, batch_size=mini_batch_size, shuffle=False)

network.eval()
with torch.no_grad():
    for data in test_load:
        x, y, z = data
        x = x.to(device)
        outputs, r = network(x)
        print(r)
        outputs = outputs[0].cpu()

        new_img_PIL = transforms.ToPILImage()(outputs).convert('RGB')
        new_img_PIL.save(path.join('./0756545', z[0]))