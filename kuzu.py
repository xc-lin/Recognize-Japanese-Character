# kuzu.py


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.layer = nn.Linear(28 * 28, 15)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        layer_out = self.layer(x)
        output = F.log_softmax(layer_out, dim=1)
        return output


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.in_to_hid = nn.Linear(28 * 28, 200)
        self.hid_to_out = nn.Linear(200, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        hid_sum = self.in_to_hid(x)
        hidden = torch.tanh(hid_sum)
        out = self.hid_to_out(hidden)
        output = F.log_softmax(out, dim=1)
        return output


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=3)
        self.maxPooling = nn.MaxPool2d(kernel_size=2)
        self.conv_to_fully = nn.Linear(1600, 400)
        self.fully_to_out = nn.Linear(400, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        conv1_sum = self.conv_layer1(x)
        conv1 = F.relu(conv1_sum)
        conv1_pooling = self.maxPooling(conv1)

        conv2_sum = self.conv_layer2(conv1_pooling)
        conv2 = F.relu(conv2_sum)
        conv2_pooling = self.maxPooling(conv2)

        shaped = self.flatten(conv2_pooling)
        # print(x.shape)
        full_sum = self.conv_to_fully(shaped)
        fully = F.relu(full_sum)

        out = self.fully_to_out(fully)
        output = F.log_softmax(out, dim=1)
        return output
