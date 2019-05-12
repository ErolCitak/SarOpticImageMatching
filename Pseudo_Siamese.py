import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
from tqdm import tqdm
from torchsummary import summary

do_learn = True
save_frequency = 2
weight_decay = 0.0001

num_epochs = 30
# 32 for positive(corresponding), 32 for negative(non-corresponding) images
batch_size = 64
lr = 0.0009
l2_lambda = 0.001


class PseudoSiamese(nn.Module):

    def __init__(self):
        super().__init__()

        ###########
        # FOR SAR IMAGE PROCESSING
        ###########

        self.s_conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.s_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.s_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.s_conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s_conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.s_conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s_conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        ###########
        # FOR OPTIC IMAGE PROCESSING
        ###########

        self.o_conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.o_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.o_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.o_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.o_conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.o_conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.o_conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.o_conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        #########
        # CONCATENATION PART
        #########
        self.dropout = nn.Dropout(0.5)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.c_conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.c_linear1 = nn.Linear(512, 512)
        self.c_linear2 = nn.Linear(512, 2)


    def forward_sar(self, x):

        x = self.s_conv1(x)
        x = F.relu(x)
        #x = nn.BatchNorm2d(x)

        x = self.s_conv2(x)
        x = F.relu(x)

        x = self.s_conv3(x)
        x = F.relu(x)

        x = self.s_conv4(x)
        x = F.relu(x)

        x = self.s_conv5(x)
        x = F.relu(x)

        x = self.s_conv6(x)
        x = F.relu(x)

        x = self.s_conv7(x)
        x = F.relu(x)

        x = self.s_conv8(x)
        x = F.relu(x)

        # x value equals to => batch_size X 2 X 2 X 128
        return x

    def forward_optic(self,x):

        x = self.o_conv1(x)
        x = F.relu(x)
        #x = nn.BatchNorm2d(x)

        x = self.o_conv2(x)
        x = F.relu(x)

        x = self.o_conv3(x)
        x = F.relu(x)

        x = self.o_conv4(x)
        x = F.relu(x)

        x = self.o_conv5(x)
        x = F.relu(x)

        x = self.o_conv6(x)
        x = F.relu(x)

        x = self.o_conv7(x)
        x = F.relu(x)

        x = self.o_conv8(x)
        x = F.relu(x)

        # x value equals to => batch_size X 2 X 2 X 128
        return x

    def concat_sar_optic(self,sarX, opticX):

        # concatenation step
        x = torch.cat((sarX, opticX), 0)

        x = self.c_conv1(x)
        x = F.relu(x)
        #x = nn.BatchNorm2d(x)

        x = self.c_conv2(x)
        x = F.relu(x)

        # Flatten
        x = x.view(x.size()[0], -1)

        # Linear layers
        x = self.c_linear1(x)
        x = self.dropout(x)
        x = self.c_linear2(x)

        return x

    def forward(self, sar_data, optic_data):

        sar_res = self.forward_sar(sar_data)
        optic_res = self.forward_sar(optic_data)

        out = self.concat_sar_optic(sar_res,optic_res)

        return out

def train(self, model, device, sar_train_loader, optic_train_loader, epoch, optimizer, mode=True):

    model.train()

    for batch_idx, (data, target) in enumerate(sar_train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)

    for batch_idx, (data, target) in enumerate(optic_train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)

    optimizer.zero_grad()

    output_positive = model(data[:2])
    output_negative = model(data[0:3:2])

    target = target.type(torch.LongTensor).to(device)
    target_positive = torch.squeeze(target[:, 0])
    target_negative = torch.squeeze(target[:, 1])

    loss_positive = F.binary_cross_entropy(output_positive, target_positive)
    loss_negative = F.binary_cross_entropy(output_negative, target_negative)

    loss = loss_positive + loss_negative

    loss.backward()

    optimizer.step()
    if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * batch_size, len(optic_train_loader.dataset),
                   100. * batch_idx * batch_size / len(optic_train_loader.dataset),
            loss.item()))

if __name__=="__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(128)])

    # load network to GPU if it's available
    network = PseudoSiamese().to("cpu")

    summary(network, input_size=[(2, 112, 112),(2, 112, 112)] , device="cpu")
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

    """
    for epoch in range(num_epochs):
        train(network, device, train_loader, epoch, optimizer)
        test(model, device, test_loader)
        if epoch & save_frequency == 0:
            torch.save(model, 'siamese_{:03}.pt'.format(epoch))
    else:  # prediction
        prediction_loader = torch.utils.data.DataLoader(
            BalancedMNISTPair('../data', train=False, download=True, transform=trans), batch_size=1, shuffle=True)
        model.load_state_dict(torch.load(load_model_path))
        data = []
        data.extend(next(iter(prediction_loader))[0][:3:2])
        same = oneshot(model, device, data)
        if same > 0:
            print('These two images are of the same number')
        else:
            print('These two images are not of the same number')
    """
