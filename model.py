import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)  # Input channels = 1, Output channels = 6, Kernel size = 5
        self.conv2 = nn.Conv2d(6, 16, 5)  # Input channels = 6, Output channels = 16, Kernel size = 5
        # Subsampling using max pooling
        self.pool = nn.MaxPool2d(2, 2)  # Kernel size = 2x2, Stride = 2
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16*4*4 from image dimension reduction from conv and pooling layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes for MNIST

    def forward(self, img):
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 60)  # First layer from 784 inputs to 60 outputs
        self.fc2 = nn.Linear(60, 40)  # Second layer from 60 inputs to 40 outputs
        self.fc3 = nn.Linear(40, 10)  # Output layer for 10 classes

    def forward(self, img):
        x = torch.flatten(img, 1)  # Flatten the image to a vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output


lenet_model = LeNet5()
mlp_model = CustomMLP()

lenet_params = sum(p.numel() for p in lenet_model.parameters() if p.requires_grad)
mlp_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)

print(f"Number of parameters in LeNet-5: {lenet_params}")
print(f"Number of parameters in Custom MLP: {mlp_params}")
