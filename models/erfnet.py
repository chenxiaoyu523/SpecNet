# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.initial_block = DownsamplerBlock(1,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        self.conv11=nn.Conv2d(1,1,2,padding=0)
    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        output=self.conv11(output)
        return output

#ERFNet
class ERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        #num_classes=1
        super().__init__()
        
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv2d(1, 16, kernel_size=(1, 104), padding=0)
       # self.conv02 = nn.Conv2d(31, 31, kernel_size=(1, 32), padding=0)
        self.conv01 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1,1))
        self.conv1 = nn.Conv2d(31, 16, kernel_size=(3, 3), padding=(1, 1))

        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=(1, 1))

        self.conv3 = nn.Conv2d(8, 1, kernel_size=(3, 3), padding=(1, 1))

        self.conv4=nn.Conv2d(1, 1, 2, stride=1, padding=1, bias=True)

        self.deconv=nn.ConvTranspose2d(16,31,kernel_size=(1, 1), padding=0)
        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        #print(input.shape)
        #print(xxxxxx)
        #plt.figure()
        #plt.imshow(input[0][0].cpu().detach().numpy(),cmap='gray')
        x=self.relu(self.conv0(input))       # old
        #print(input.shape, x.shape)
       # plt.figure()
       # plt.imshow(x[0][0].cpu().detach().numpy(),cmap='gray')
       # plt.show()
       # x=torch.nn.functional.interpolate(x,[127,127],mode='bilinear')
       # print(x.shape)
       # print(xxxxxx)
        
        x=self.deconv(x)
        
        #print(x.shape)
        #print(xxxxxx)

    #    x=self.deconv(input)
     #   x=self.relu(self.conv02(x))

        #print(xxxxxx)
        x=self.relu(self.conv1(x))
        #print(x.shape)
        #print(xxxxxx)
        #x=torch.Tensor(1,16,127,127).cuda()
        x=self.relu(self.conv2(x))
        x=self.relu(self.conv3(x))
        #y=x.squeeze(0).detach().cpu().numpy()
        x=self.conv4(x)
        x=torch.nn.functional.interpolate(x,[128,128],mode='bilinear', align_corners=True)
      #  plt.figure()
      #  plt.imshow(x[0][0].cpu().detach().numpy(),cmap='gray')
      #  plt.show()
        #y=x.squeeze(0).squeeze(0).detach().cpu().numpy()
        #cv.imshow('222',y[2,:,:])
        #cv.imshow('333',y[3,:,:])
        #cv.imshow('444',y[4,:,:])
        #print(xxxxxx)
       
        if only_encode:
            return self.encoder.forward(x, predict=True)
        else:
            output = self.encoder(x)    #predict=False by default
            return self.decoder.forward(output)
