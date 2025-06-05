import torch.nn as nn
from torchvision import models

class BrainTumorSqueezeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(BrainTumorSqueezeNet, self).__init__()
        self.model = models.squeezenet1_1(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        self.model.num_classes = num_classes

    def forward(self, x):
        return self.model(x)
