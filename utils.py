import matplotlib.pyplot as plt
import json

def plot_training_history():
    with open('training_logs.json', 'r') as f:
        logs = json.load(f)
    
    plt.figure(figsize=(10, 6))
    plt.plot(logs['epochs'], logs['train_loss'], label='Training Loss')
    plt.plot(logs['epochs'], logs['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('static/training_history.png')
    plt.close() 