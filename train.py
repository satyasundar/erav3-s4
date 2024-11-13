import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
from model import MNISTNet
from tqdm import tqdm
import logging
from state import model_plots, training_status

logger = logging.getLogger(__name__)

def get_optimizer(name, parameters, lr):
    if name.lower() == 'adam':
        return optim.Adam(parameters, lr=lr)
    elif name.lower() == 'sgd':
        return optim.SGD(parameters, lr=lr)
    elif name.lower() == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def start_training(config, training_status, model_name):
    try:
        logger.info(f"Starting training for model: {model_name}")
        
        # Initialize model data in model_plots
        model_plots[model_name] = {
            'train_loss': [], 
            'val_loss': [], 
            'train_acc': [], 
            'val_acc': [], 
            'epochs': [], 
            'config': config,
            'status': 'running'
        }
        
        # Data preparation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

        # Initialize device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize model with config
        model = MNISTNet(config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(config['optimizer'], model.parameters(), config['learning_rate'])

        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct/total:.2f}%'
                })
            
            train_accuracy = 100. * correct / total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_accuracy = 100. * correct / total
            avg_val_loss = val_loss / len(test_loader)
            
            # Update model plots
            model_plots[model_name]['train_loss'].append(avg_train_loss)
            model_plots[model_name]['val_loss'].append(avg_val_loss)
            model_plots[model_name]['train_acc'].append(train_accuracy)
            model_plots[model_name]['val_acc'].append(val_accuracy)
            model_plots[model_name]['epochs'].append(epoch)
            
            # Save training logs after each epoch
            with open('training_logs.json', 'w') as f:
                json.dump(model_plots, f)
            
            logger.info(f"Epoch {epoch+1} completed - Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
        
        # Update status when training completes
        model_plots[model_name]['status'] = 'completed'
        training_status['status'] = 'completed'
        logger.info("Training completed successfully")
        
        # Final save of training logs
        with open('training_logs.json', 'w') as f:
            json.dump(model_plots, f)
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        training_status['status'] = 'error'
        training_status['error'] = str(e)
        if model_name in model_plots:
            model_plots[model_name]['status'] = 'error'