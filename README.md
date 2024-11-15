# MNIST CNN Classifier with Real-time Training Visualization

A 3-layer Convolutional Neural Network trained on the MNIST dataset with real-time training visualization through a web interface.

## Features
- 3-layer CNN architecture
- Customizable parameters: 
    - Batch size
    - Learning rate 
    - Number of epochs 
    - Optimizer
    - Model architecture parameters: 
        - Layers(16-32-64 or 8-8-8 etc) 
        - Kernel size(3 or 5 etc)
        - Activation function(ReLU or Sigmoid etc) 
        - Pooling layer(Max or Avg etc)
- Real-time training loss and accuracy visualization on any number of models OVERLAPPING on the same plot
- Test results on random MNIST samples

- Clear model card button to remove previous model card, Clear predictions button to remove previous predictions, 
- Progress bar for training epochs with elapsed time and accuracy
- CUDA/CPU/MPS training acceleration (CUDA is for NVIDIA GPUs, MPS is for Apple Silicon GPUs)

## Project Structure
- `model.py`: Contains the CNN architecture
- `train.py`: Training script with logging functionality
- `server.py`: Simple HTTP server for real-time visualization
- `templates/index.html`: Web interface for visualization
- `utils.py`: Utility functions for data handling and visualization

## Requirements
- PyTorch
- Flask
- Matplotlib
- NumPy
- torchvision

## Setup and Running
1. Install requirements:
   ```bash
   mkdir static templates
   pip install torch torchvision flask matplotlib numpy
   ```

2. Start the visualization server:
   ```bash
   python server.py
   ```


3. Open browser and go to:
   ```
   http://127.0.0.1:5000
   ```
