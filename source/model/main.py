import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
from random import randint

def read_img(size):
    # Read in images, return 2 numpy array
    original_path = "../../resource/training_data_set/colored_data_set"
    lineart_path = "../../resource/training_data_set/sketch_data_set"


    original_files = sorted(os.listdir(original_path))
    lineart_files = sorted(os.listdir(lineart_path))

    X_o = []
    X_l = []

    for i,img in enumerate(lineart_files):
        if i < size:
            img_lineart_path = os.path.join(lineart_path, img)
            img_original_path = os.path.join(original_path, img)

            img_lineart = cv2.imread(img_lineart_path)
            img_original = cv2.imread(img_original_path)

            img_original = cv2.resize(img_original, (512,512))
            img_lineart = cv2.resize(img_lineart, (512,512))
            img_lineart = cv2.cvtColor(img_lineart, cv2.COLOR_RGB2GRAY)

            X_o.append(img_original)
            X_l.append(img_lineart)
        else:
            break

    X_o = np.array(X_o)
    X_l = np.array(X_l)
    return X_o, X_l

def imageblur(cimg, sampling=False):
    if sampling:
        cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
    else:
        for i in range(15):
            randx = randint(0, 205)
            randy = randint(0, 205)
            cimg[randx:randx + 50, randy:randy + 50] = 255
    return cv2.blur(cimg, (100, 100))

def show_img(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def flatten(x):
    N = x.shape[0]
    return x.view(N,-1)

class Model():
    def __init__(self, batchsize=4, img_size=512):
        self.batchsize = batchsize
        self.outputsize = img_size
        self.inputsize = img_size
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5,stride=2,padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(128)
        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=2, output_padding=(1, 1))
        self.dconv2 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2, padding=2, output_padding=(1, 1))
        self.dconv3 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=2, output_padding=(1, 1))
        self.dconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=2, output_padding=(1, 1))
        self.dconv5 = nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2, padding=2, output_padding=(1, 1))

        self.dbn1 = nn.BatchNorm2d(128)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(7, 64, kernel_size=5, stride=2, padding=2)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(32*32*32, 1)

    def discriminator(self, img):
        ''' img size is (512,512,4+3) '''
        h0 = self.lrelu(self.conv6(img))  # (512,512,7)->(256,256,64)
        h1 = self.lrelu(self.dbn1(self.conv7(h0)))  # (256,256,64)->(128,128,128)
        h2 = self.lrelu(self.dbn2(self.conv8(h1)))  # (128,128,128)->(64,64,256)
        h3 = self.lrelu(self.dbn3(self.conv9(h2)))  # (64,64,256)->(32,32,512)
        h4 = self.fc(flatten(h3))  # (32,32,512)->(1)
        return nn.Sigmoid(h4), h4

    def generator(self, img):
        e1 = self.conv1(img)  # (512,512,4)->(256,256,64)
        e2 = self.bn2(self.conv2(self.lrelu(e1)))  # (256,256,64)->(128,128,128)
        e3 = self.bn3(self.conv3(self.lrelu(e2)))  # (128,128,128)->(64,64,256)
        e4 = self.bn4(self.conv4(self.lrelu(e3)))  # (64,64,256)->(32,32,512)
        e5 = self.bn5(self.conv5(self.lrelu(e4)))  # (32,32,512)->(16,16,1024)

        d4 = self.dconv1(self.relu(e5))  # (16,16,1024)->(32,32,512)
        d4 = self.bn6(d4)
        d4 = torch.cat([d4, e4], 3)  # (32,32,1024)

        d3 = self.dconv2(self.relu(d4))  # (32,32,1024)->(64,64,256)
        d3 = self.bn7(d3)
        d3 = torch.cat([d3, e3], 3)  # (64,64,512)

        d2 = self.dconv3(self.relu(d3))  # (64,64,512)->(128,128,128)
        d2 = self.bn8(d2)
        d2 = torch.cat([d2, e2], 3)  # (128,128,256)

        d1 = self.dconv4(self.relu(d2))  # (128,128,256)->(256,256,64)
        d1 = self.bn1(d1)
        d1 = torch.cat([d1, e1], 3)  # (256,256,128)

        d0 = self.dconv5(self.relu(d1))  # (256,256,128)->(512,512,3)

        return F.tanh(d0)  # (512,512,3)

    def train(self):







X_o,X_l = read_img(20)
cimg = X_o[4]
show_img(cimg)
blur = imageblur(cimg)
show_img(blur)






