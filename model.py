import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Corrected input size for the fully connected layer
        # After 4 max pooling operations: 28x28 -> 14x14 -> 7x7 -> 3x3 -> 1x1
        # Final feature map will be 128 channels * 1 * 1
        self._to_linear = 128  # Corrected size
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self._to_linear, 10)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        x = self.pool(F.relu(self.conv4(x)))  # 3x3 -> 1x1
        
        # Flatten while preserving batch size
        x = x.view(batch_size, -1)  # Should be [batch_size, 128]
        x = self.dropout(x)
        x = self.fc1(x)
        return x