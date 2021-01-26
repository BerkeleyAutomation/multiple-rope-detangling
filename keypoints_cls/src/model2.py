import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/host/src')

class Resnet4Channel(nn.Module):
        def __init__(self,num_keypoints=1, num_classes=1, num_channels=3, pretrained=False,img_height=480, img_width=640):
                super(Resnet4Channel, self).__init__()
                self.num_keypoints = num_keypoints
                self.num_outputs = self.num_keypoints
                self.img_height = img_height
                self.img_width = img_width
                self.resnet = torchvision.models.resnet34(pretrained=False, num_classes=num_classes)
                #self.resnet = Resnet34_8s(channels=4, pretrained=False)
                self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.sigmoid = nn.Sigmoid()
        def forward(self, x):
                output = self.resnet(x)
                output = self.sigmoid(output)
                return output

if __name__ == '__main__':
        model = KeypointsGauss(4).cuda()
        x = torch.rand((1,3,480,640)).cuda()
        result = model.forward(x)
        print(x.shape)
        print(result.shape)
