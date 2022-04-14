import torch.nn.functional as F
from torch import nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(6, 256, 2, 2)
       # self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(256, 512, 2, 2)
        # self.batch_norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(512, 512, 2, 2)
        # self.batch_norm3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # x = self.batch_norm1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Use the rectified-linear activation function over x

        x = self.conv2(x)
        # x = self.batch_norm2(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = F.relu(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        # try to add relu to linear layer
        x = self.fc1(x)
        # x = self.fc3(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output

input = torch.randn(size=(1, 6, 15, 15))
net = Net()
out = net.forward(input)
print(out.shape)
