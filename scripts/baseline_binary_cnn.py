import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    Classic LeNet CNN-Model for binary Classification. Takes grayscale images with shape (32x32) as input
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BaselineNet(nn.Module):
    """
    Baseline CNN-Model for binary classification. Takes grayscale images with shape (254,254) as input
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.pool1 = nn.MaxPool2d()
        self.pool2 = nn.MaxPool2d()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()


    def forward(self, x):
        x = self.pool1 =(F.relu(self.conv1(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x