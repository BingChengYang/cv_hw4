import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transform
import matplotlib
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse
# my own library
import load_data_1
import net

model = "model300.pkl"
lr = 0.1
epoches = 2000
mini_batch_size = 64
step = 500
load_model = False
adjust_lr = True
crop_size = 41

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1)
parser.add_argument("--mini_batch_size", default=64)
parser.add_argument("--epoches", default=2000)
parser.add_argument("--step", default=500)
parser.add_argument("--load_model", default=False)
parser.add_argument("--adjust_lr", default=True)
parser.add_argument("--crop_size", default=41)
parser.add_argument("--model", default="model.pkl")
args = parser.parse_args()

hr_transform = transform.Compose([
    transform.ToTensor(),
])

img_transform = transform.Compose([
    transform.RandomCrop((args.crop_size, args.crop_size)),
    transform.RandomHorizontalFlip(),
    transform.RandomVerticalFlip(),
    transform.RandomRotation(0, 360),
    transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,hue=0.5)
])

if load_model == True:
    network = torch.load('./model/'+args.model)
else:
    network = net.Net()
print(network)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
network.to(device)

train_set = load_data_1.Load_traindata(transform=img_transform, hr_transform=hr_transform, crop_size=args.crop_size)
train_load = Data.DataLoader(dataset=train_set, batch_size=args.mini_batch_size, shuffle=True)

optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
loss_func = nn.MSELoss()
scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

for epoch in range(args.epoches):
    print(epoch)
    network.train()
    if epoch > 3*step:
        adjust_lr = False
    if adjust_lr == True:
        scheduler.step()
    for step, (x, y) in enumerate(train_load):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, r_out = network(x)
        r = y - x
        loss = loss_func(r, r_out)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 0.4) 
        optimizer.step()
    print(loss)
    if (epoch % 20) == 0:
        torch.save(network, './model/model{}.pkl'.format(epoch))
# save model
torch.save(network, './model/model.pkl')