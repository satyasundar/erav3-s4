import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, config):
        super(MNISTNet, self).__init__()
        self.config = config
        
        # Get activation function
        self.activation = self._get_activation()
        
        # Define layers
        self.conv1 = nn.Conv2d(1, config['num_filters'], kernel_size=config['kernel_size'], padding=config['kernel_size']//2)
        self.conv2 = nn.Conv2d(config['num_filters'], config['num_filters']*2, kernel_size=config['kernel_size'], padding=config['kernel_size']//2)
        self.conv3 = nn.Conv2d(config['num_filters']*2, config['num_filters']*2, kernel_size=config['kernel_size'], padding=config['kernel_size']//2)
        self.conv4 = nn.Conv2d(config['num_filters']*2, config['num_filters']*4, kernel_size=config['kernel_size'], padding=config['kernel_size']//2)
        
        # Set pooling layer
        self.pool = nn.MaxPool2d(2, 2) if config['pooling'] == 'max' else nn.AvgPool2d(2, 2)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(config['num_filters']*4, 10)
        
    def _get_activation(self):
        if self.config['activation'] == 'relu':
            return F.relu
        elif self.config['activation'] == 'tanh':
            return torch.tanh
        elif self.config['activation'] == 'sigmoid':
            return torch.sigmoid
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = self.pool(self.activation(self.conv4(x)))
        
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x