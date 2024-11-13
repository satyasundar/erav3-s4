import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
from model import MNISTNet
import random
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Reduce batch size and ensure drop_last is True for both loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

# Initialize device (MPS for Mac, CUDA for NVIDIA GPU, CPU as fallback)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training logs
training_logs = {'train_loss': [], 'val_loss': [], 'epochs': []}

def train(epochs):
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Training loop with tqdm
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{100. * correct/total:.2f}%'
            })
        
        train_accuracy = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        # Validation loop with tqdm
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{val_loss/len(test_loader):.4f}',
                    'accuracy': f'{100. * correct/total:.2f}%'
                })
        
        val_accuracy = 100. * correct / total
        avg_val_loss = val_loss / len(test_loader)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training    - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Validation  - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')
        
        # Update logs
        training_logs['train_loss'].append(avg_train_loss)
        training_logs['val_loss'].append(avg_val_loss)
        training_logs['epochs'].append(epoch)
        
        # Save logs
        with open('training_logs.json', 'w') as f:
            json.dump(training_logs, f)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_accuracy:.2f}%\n')

def test_random_samples():
    model.eval()
    # Get 10 random test samples
    indices = random.sample(range(len(test_dataset)), 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            data, target = test_dataset[sample_idx]
            output = model(data.unsqueeze(0).to(device))
            pred = output.argmax(dim=1, keepdim=True)[0]
            
            # Plot image and prediction
            axes[idx].imshow(data.squeeze(), cmap='gray')
            axes[idx].set_title(f'Pred: {pred.item()}\nTrue: {target}')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('static/test_results.png')
    plt.close()

if __name__ == '__main__':
    train(epochs=10)
    test_random_samples() 