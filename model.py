import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, config):
        super(MNISTNet, self).__init__()
        self.config = config
        
        # Get activation function
        self.activation = self._get_activation()
        
        # Define layers with individual filter counts
        self.conv1 = nn.Conv2d(1, config['num_filters_1'], kernel_size=config['kernel_size'], 
                              padding=config['kernel_size']//2)
        self.conv2 = nn.Conv2d(config['num_filters_1'], config['num_filters_2'], 
                              kernel_size=config['kernel_size'], padding=config['kernel_size']//2)
        self.conv3 = nn.Conv2d(config['num_filters_2'], config['num_filters_3'], 
                              kernel_size=config['kernel_size'], padding=config['kernel_size']//2)
        
        # Set pooling layer
        self.pool = nn.MaxPool2d(2, 2) if config['pooling'] == 'max' else nn.AvgPool2d(2, 2)
        
        # Calculate the size of flattened features after convolutions and pooling
        # Input image is 28x28, after 3 pooling layers it becomes 3x3
        # So the flattened size will be: num_filters_3 * 3 * 3
        self.flat_features = config['num_filters_3'] * 3 * 3
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.flat_features, 10)
        
    def _get_activation(self):
        if self.config['activation'] == 'relu':
            return F.relu
        elif self.config['activation'] == 'tanh':
            return torch.tanh
        elif self.config['activation'] == 'sigmoid':
            return torch.sigmoid
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.pool(self.activation(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.activation(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(self.activation(self.conv3(x)))  # 7x7 -> 3x3
        
        x = x.view(batch_size, -1)  # Flatten: (batch_size, num_filters_3 * 3 * 3)
        x = self.dropout(x)
        x = self.fc1(x)
        return x