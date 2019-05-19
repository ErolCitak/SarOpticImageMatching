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
weight_decay = 0.001 # 0.001 in paper

num_epochs = 30
# 40 for positive(corresponding), 40 for negative(non-corresponding) images
batch_size = 32
lr = 0.0009
l2_lambda = 0.001

def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class HDF5Dataset(data.Dataset):

    def __init__(self, dataset_path):
        super(HDF5Dataset, self).__init__()

        print("Initialization step")

        # Read whole h5 file
        self.hf = h5.File(dataset_path, 'r')

        print("Keys of h5 store file: ", self.hf.keys())

        # Get each branch's elements and also label
        self.sar_pos_group = self.hf.get('sar_pos_group')
        self.optic_pos_group = self.hf.get('optic_pos_group')
        self.labels_pos = self.hf.get('labels_pos')

        # Get each branch's elements and also label
        self.sar_neg_group = self.hf.get('sar_neg_group')
        self.optic_neg_group = self.hf.get('optic_neg_group')
        self.labels_neg = self.hf.get('labels_neg')

        # transformation definition
        self.transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __getitem__(self, index):

        siamese_pos_label = np.asarray(self.labels_pos[index])
        siamese_neg_label = np.asarray(self.labels_neg[index])

        # Read each image and Convert image from numpy array to PIL image, mode 'L' is for grayscale
        sar_pos_img = np.asarray(self.sar_pos_group[index]).astype(np.uint8)
        optic_pos_img = np.asarray(self.optic_pos_group[index]).astype(np.uint8)

        sar_neg_img = np.asarray(self.sar_neg_group[index]).astype(np.uint8)
        optic_neg_img = np.asarray(self.optic_neg_group[index]).astype(np.uint8)

        # Array to PIL Image
        sar_pos_img = Image.fromarray(sar_pos_img)
        optic_pos_img = Image.fromarray(optic_pos_img)

        sar_neg_img = Image.fromarray(sar_neg_img)
        optic_neg_img = Image.fromarray(optic_neg_img)

        # Apply pre-defined transformations
        sar_pos_img = self.transformations(sar_pos_img)
        optic_pos_img = self.transformations(optic_pos_img)
        siamese_pos_label = torch.from_numpy(siamese_pos_label)

        sar_neg_img = self.transformations(sar_neg_img)
        optic_neg_img = self.transformations(optic_neg_img)
        siamese_neg_label = torch.from_numpy(siamese_neg_label)

        # Return images and corresponding label
        return (sar_pos_img, optic_pos_img, siamese_pos_label, sar_neg_img, optic_neg_img, siamese_neg_label)

    def __len__(self):
        # multiplied by 2 since we have pos and negative image pairs
        # TO-DO!!
        return len(self.sar_pos_group)


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
        self.dropout = nn.Dropout(0.7)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)

        self.batch_norm_32 = nn.BatchNorm2d(32)
        self.batch_norm_64 = nn.BatchNorm2d(64)
        self.batch_norm_128 = nn.BatchNorm2d(128)
        self.batch_norm_256 = nn.BatchNorm2d(256)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.c_conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.c_linear1 = nn.Linear(512, 512)
        self.c_linear2 = nn.Linear(512, 2)


    def forward_sar(self, x):


        x = self.s_conv1(x)
        x = F.relu(x)
        x = self.batch_norm_32(x)

        x = self.dropout3(x)

        x = self.s_conv2(x)
        x = F.relu(x)
        x = self.batch_norm_32(x)

        x = self.max_pooling(x)

        x = self.s_conv3(x)
        x = F.relu(x)
        x = self.batch_norm_64(x)

        x = self.dropout5(x)

        x = self.s_conv4(x)
        x = F.relu(x)
        x = self.batch_norm_64(x)

        x = self.max_pooling(x)

        x = self.s_conv5(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)

        x = self.s_conv6(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)

        x = self.max_pooling(x)

        """
        x = self.s_conv7(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)

        x = self.s_conv8(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)
        """
        # x value equals to => batch_size X 2 X 2 X 128
        return x

    def forward_optic(self,x):

        x = self.o_conv1(x)
        x = F.relu(x)
        x = self.batch_norm_32(x)

        x = self.dropout3(x)

        x = self.o_conv2(x)
        x = F.relu(x)
        x = self.batch_norm_32(x)

        x = self.max_pooling(x)

        x = self.o_conv3(x)
        x = F.relu(x)
        x = self.batch_norm_64(x)

        x = self.o_conv4(x)
        x = F.relu(x)
        x = self.batch_norm_64(x)

        x = self.max_pooling(x)

        x = self.o_conv5(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)

        x = self.o_conv6(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)

        x = self.max_pooling(x)

        """
        x = self.o_conv7(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)

        x = self.o_conv8(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)
        """
        # x value equals to => batch_size X 2 X 2 X 128
        return x

    def concat_sar_optic(self,sarX, opticX):

        # concatenation step
        x = torch.cat((sarX, opticX), dim=1)

        x = self.c_conv1(x)
        x = F.relu(x)
        x = self.batch_norm_256(x)

        x = self.c_conv2(x)
        x = F.relu(x)
        x = self.batch_norm_128(x)

        x = self.max_pooling(x)

        # Flatten
        x = x.view(x.size()[0], -1)

        # Linear layers
        x = self.c_linear1(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.c_linear2(x)

        return x

    def forward(self, sar_data, optic_data):

        sar_res = self.forward_sar(sar_data)
        optic_res = self.forward_optic(optic_data)

        out = self.concat_sar_optic(sar_res,optic_res)
        out = F.softmax(out)

        return out

def train(model, device, data_loader, epoch, optimizer):

    loss_f = nn.BCELoss()

    for batch_idx, (sar_pos_data, optic_pos_data, target_pos, sar_neg_data, optic_neg_data, target_neg) in enumerate(data_loader):

        """
        for i in range(len(sar_pos_data)):
            sar_pos_data[i] = sar_pos_data[i].to(device)
            optic_pos_data[i] = optic_pos_data[i].to(device)
            target_pos[i] = target_pos[i].to(device)

            sar_neg_data[i] = sar_neg_data[i].to(device)
            optic_neg_data[i] = optic_neg_data[i].to(device)
            target_neg[i] = target_neg[i].to(device)
        """
        optimizer.zero_grad()

        model.train()

        output_positive = model(sar_pos_data.to(device), optic_pos_data.to(device))

        output_negative = model(sar_neg_data.to(device), optic_neg_data.to(device))

        target_pos = target_pos.to(device, dtype=torch.float32)
        target_positive = torch.squeeze(target_pos)

        target_neg = target_neg.to(device, dtype=torch.float32)
        target_negative = torch.squeeze(target_neg)

        l2_reg = 0
        for W in model.parameters():
            l2_reg = l2_reg + W.norm(2)


        loss_positive = loss_f(output_positive, target_positive)
        loss_negative = loss_f(output_negative, target_negative)
        #print("Loss Pos/Neg:", loss_positive, loss_negative)
        #loss_negative = F.binary_cross_entropy(output_negative, target_negative)

        #loss = loss_positive
        loss = loss_positive + loss_negative + (l2_reg * l2_lambda)

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(data_loader.dataset),
                       100. * batch_idx * batch_size / len(data_loader.dataset),
                loss.item()))


def test(model, device, test_loader):
    model.eval()
    loss_f = nn.BCELoss()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0
        for batch_idx, (sar_pos_data, optic_pos_data, target_pos, sar_neg_data, optic_neg_data, target_neg) in enumerate(test_loader):

            """
            for i in range(len(sar_pos_data)):
                sar_pos_data[i] = sar_pos_data[i].to(device)
                optic_pos_data[i] = optic_pos_data[i].to(device)
                target_pos[i] = target_pos[i].to(device)

                sar_neg_data[i] = sar_neg_data[i].to(device)
                optic_neg_data[i] = optic_neg_data[i].to(device)
                target_neg[i] = target_neg[i].to(device)
            """
            output_positive = model(sar_pos_data.to(device), optic_pos_data.to(device))
            output_negative = model(sar_neg_data.to(device), optic_neg_data.to(device))

            # Ground truth operations
            target_pos = target_pos.to(device, dtype=torch.float32)
            target_positive = torch.squeeze(target_pos)

            target_neg = target_neg.to(device, dtype=torch.float32)
            target_negative = torch.squeeze(target_neg)

            loss_positive = loss_f(output_positive, target_positive)
            loss_negative = loss_f(output_negative, target_negative)

            loss = loss + loss_positive + loss_negative

            accurate_labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == torch.argmax(target_positive, dim=1)).cpu()
            accurate_labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == torch.argmax(target_negative, dim=1)).cpu()

            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))


if __name__=="__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load network to GPU if it's available
    network = PseudoSiamese().to(device)
    network.apply(init_xavier)

    """
    summary(network, input_size=[(2, 128, 128),(2, 128, 128)] , device="cpu")
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    """

    # Call dataset
    Train_Dataset =  HDF5Dataset('D:/Sen1_2_/Train_Matching.h5')
    Test_Dataset =  HDF5Dataset('D:/Sen1_2_/Validation_Matching.h5')

    # Define data loader
    siamese_train_loader = torch.utils.data.DataLoader(dataset=Train_Dataset, batch_size=batch_size)
    siamese_test_loader = torch.utils.data.DataLoader(dataset=Test_Dataset, batch_size=batch_size)

    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

    # Dataset Infos
    """
    for sar, optic, label in siamese_dataset_loader:
        print("Sar Shape:",sar.shape)
        print("Optic Shape:",optic.shape)
        print("Label:",label)
    """

    for epoch in range(num_epochs):
        train(network, device, siamese_train_loader, epoch, optimizer)
        test(network,device,siamese_test_loader)

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
