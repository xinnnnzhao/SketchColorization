import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)
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
        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=(1, 1))
        self.dconv2 = nn.ConvTranspose2d(1024, 256, kernel_size=5, stride=2, padding=2, output_padding=(1, 1))
        self.dconv3 = nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=(1, 1))
        self.dconv4 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2, output_padding=(1, 1))
        self.dconv5 = nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, padding=2, output_padding=(1, 1))

    def forward(self, X):
        e1 = self.conv1(X)  # (512,512,4)->(256,256,64)
        e2 = self.bn2(self.conv2(self.lrelu(e1)))  # (256,256,64)->(128,128,128)
        e3 = self.bn3(self.conv3(self.lrelu(e2)))  # (128,128,128)->(64,64,256)
        e4 = self.bn4(self.conv4(self.lrelu(e3)))  # (64,64,256)->(32,32,512)
        e5 = self.bn5(self.conv5(self.lrelu(e4)))  # (32,32,512)->(16,16,1024)

        d4 = self.dconv1(self.relu(e5))  # (16,16,1024)->(32,32,512)
        d4 = self.bn6(d4)
        d4 = torch.cat([d4, e4], 1)  # (32,32,1024)

        d3 = self.dconv2(self.relu(d4))  # (32,32,1024)->(64,64,256)
        d3 = self.bn7(d3)
        d3 = torch.cat([d3, e3], 1)  # (64,64,512)

        d2 = self.dconv3(self.relu(d3))  # (64,64,512)->(128,128,128)
        d2 = self.bn8(d2)
        d2 = torch.cat([d2, e2], 1)  # (128,128,256)

        d1 = self.dconv4(self.relu(d2))  # (128,128,256)->(256,256,64)
        d1 = self.bn1(d1)
        d1 = torch.cat([d1, e1], 1)  # (256,256,128)

        d0 = self.dconv5(self.relu(d1))  # (256,256,128)->(512,512,3)

        return F.tanh(d0)  # (512,512,3)

    def save(self):
        torch.save(self.state_dict(), "../../resource/model/generator.md")

    def load(self):
        self.load_state_dict(torch.load("../../resource/model/generator.md"))
