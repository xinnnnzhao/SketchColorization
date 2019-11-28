import torch
from torch import nn
import torch.nn.functional as F
from .models import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dbn1 = nn.BatchNorm2d(128)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(7, 64, kernel_size=5, stride=2, padding=2)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(32 * 32 * 512, 1)

    def forward(self, X):
        ''' img size is (512,512,4+3) '''
        h0 = self.lrelu(self.conv6(X))  # (512,512,7)->(256,256,64)
        h1 = self.lrelu(self.dbn1(self.conv7(h0)))  # (256,256,64)->(128,128,128)
        h2 = self.lrelu(self.dbn2(self.conv8(h1)))  # (128,128,128)->(64,64,256)
        h3 = self.lrelu(self.dbn3(self.conv9(h2)))  # (64,64,256)->(32,32,512)
        h4 = self.fc(flatten(h3))  # (32,32,512)->(1)
        return F.sigmoid(h4)

    def save(self):
        torch.save(self.state_dict(), "../../resource/model/discriminator.md")

    def load(self):
        self.load_state_dict(torch.load("../../resource/model/discriminator.md"))
