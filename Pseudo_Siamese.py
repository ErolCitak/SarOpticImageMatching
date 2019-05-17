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
from torch.utils import data
from torchvision import transforms

import h5py as h5

do_learn = True
save_frequency = 2
weight_decay = 0.0001

num_epochs = 30
# 32 for positive(corresponding), 32 for negative(non-corresponding) images
batch_size = 1
lr = 0.0009
l2_lambda = 0.001


class HDF5Dataset(data.Dataset):

    def __init__(self, dataset_path):
        print("Initialization step")

        # Read whole h5 file
        self.hf = h5.File(dataset_path, 'r')

        print("Keys of h5 store file: ", self.hf.keys())

        # Get each branch's elements and also label
        self.sar_group = self.hf.get('sar_group')
        self.optic_group = self.hf.get('optic_group')
        self.labels = self.hf.get('labels')

        # transformation definition
        self.transformations = transforms.Compose([transforms.ToTensor()])


    def __getitem__(self, index):

        siamese_label = np.asarray(self.labels[index])

        # Read each image and Convert image from numpy array to PIL image, mode 'L' is for grayscale
        sar_img = np.asarray(self.sar_group[index]).astype('uint8')
        optic_img = np.asarray(self.optic_group[index]).astype('uint8')

        sar_img = Image.fromarray(sar_img)
        optic_img = Image.fromarray(optic_img)

        sar_img = sar_img.convert('L')
        optic_img = optic_img.convert('L')

        # Apply pre-defined transformations
        sar_img = self.transformations(sar_img)
        optic_img = self.transformations(optic_img)
        siamese_label = torch.from_numpy(siamese_label)

        # Return images and corresponding label
        return (sar_img, optic_img, siamese_label)

    def __len__(self):
        return len(self.sar_group)

class PseudoSiamese(nn.Module):

    def __init__(self):
        super().__init__()

        ###########
        # FOR SAR IMAGE PROCESSING
        ###########

        self.s_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
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

        self.o_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
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

        x = self.s_conv2(x)
        x = F.relu(x)
        x = self.max_pooling(x)

        x = self.s_conv3(x)
        x = F.relu(x)

        x = self.s_conv4(x)
        x = F.relu(x)
        x = self.max_pooling(x)

        x = self.s_conv5(x)
        x = F.relu(x)

        x = self.s_conv6(x)
        x = F.relu(x)
        x = self.max_pooling(x)

        x = self.s_conv7(x)
        x = F.relu(x)

        x = self.s_conv8(x)
        x = F.relu(x)

        # x value equals to => batch_size X 2 X 2 X 128
        return x

    def forward_optic(self,x):

        x = self.o_conv1(x)
        x = F.relu(x)

        x = self.o_conv2(x)
        x = F.relu(x)
        x = self.max_pooling(x)

        x = self.o_conv3(x)
        x = F.relu(x)

        x = self.o_conv4(x)
        x = F.relu(x)
        x = self.max_pooling(x)


        x = self.o_conv5(x)
        x = F.relu(x)

        x = self.o_conv6(x)
        x = F.relu(x)
        x = self.max_pooling(x)

        x = self.o_conv7(x)
        x = F.relu(x)

        x = self.o_conv8(x)
        x = F.relu(x)

        # x value equals to => batch_size X 2 X 2 X 128
        return x

    def concat_sar_optic(self,sarX, opticX):

        # concatenation step
        x = torch.cat((sarX, opticX), dim=1)

        x = self.c_conv1(x)
        x = F.relu(x)
        #x = nn.BatchNorm2d(x)

        x = self.c_conv2(x)
        x = F.relu(x)
        x = self.max_pooling(x)

        # Flatten
        x = x.view(x.size()[0], -1)

        # Linear layers
        x = self.c_linear1(x)
        x = self.dropout(x)
        x = self.c_linear2(x)

        return x

    def forward(self, sar_data, optic_data):

        sar_res = self.forward_sar(sar_data)
        optic_res = self.forward_optic(optic_data)

        out = self.concat_sar_optic(sar_res,optic_res)
        out = F.sigmoid(out)
        # TO.DO. -> Cross-entropy
        ####
        ####
        ####

        return out

def train(model, device, data_loader, epoch, optimizer):

    model.train()
    loss_f = nn.BCELoss()

    for batch_idx, (sar_data, optic_data, target) in enumerate(data_loader):
        for i in range(len(sar_data)):
            sar_data[i] = sar_data[i].to(device)
            optic_data[i] = optic_data[i].to(device)


    optimizer.zero_grad()

    output_positive = model(sar_data,optic_data)
    #output_positive = torch.squeeze(output_positive)
    #output_negative = model(data[0:3:2])

    target = target.to(device,dtype=torch.float32)
    target_positive = (target)
    target_positive = torch.squeeze(target)

    print("############")
    print("Output Positive:", output_positive)
    print("Output Positive Shape:", output_positive.shape)
    print("Target Positive:", target_positive)
    print("Target Shape w/ Squeeze:", target_positive.shape)
    print("############")

    #target_negative = torch.squeeze(target[:, 1])

    loss_positive = loss_f(output_positive, target_positive)
    #loss_negative = F.binary_cross_entropy(output_negative, target_negative)

    loss = loss_positive
    #loss = loss_positive + loss_negative

    loss.backward()

    optimizer.step()
    if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * batch_size, len(data_loader.dataset),
                   100. * batch_idx * batch_size / len(data_loader.dataset),
            loss.item()))

if __name__=="__main__":

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    # load network to GPU if it's available
    network = PseudoSiamese().to("cpu")

    summary(network, input_size=[(1, 128, 128),(1, 128, 128)] , device="cpu")
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)


    # Call dataset
    Train_Dataset =  HDF5Dataset('./dataset/Train_Matching.h5')
    print("Hello Dude")
    # Define data loader
    siamese_dataset_loader = torch.utils.data.DataLoader(dataset=Train_Dataset,
                                                    batch_size=1)

    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

    # Dataset Infos

    for sar, optic, label in siamese_dataset_loader:
        print("Sar Shape:",sar.shape)
        print("Optic Shape:",optic.shape)
        print("Label:",label)


    for epoch in range(num_epochs):
        train(network, device, siamese_dataset_loader, epoch, optimizer)


    # Complete train-test
    """
    for epoch in range(num_epochs):
        train(network, device, siamese_dataset_loader, epoch, optimizer)
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
