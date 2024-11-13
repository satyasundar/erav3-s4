# MNIST CNN Classifier with Real-time Training Visualization

A 4-layer Convolutional Neural Network trained on the MNIST dataset with real-time training visualization through a web interface.

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

3. In a new terminal, start training:
   ```bash
   python train.py
   ```

4. Open browser and go to:
   ```
   http://localhost:5000
   ```

## Features
- 4-layer CNN architecture
- Real-time training loss visualization
- Test results on random MNIST samples
- Web-based monitoring interface

## Model Architecture
- Conv1: 1 → 32 channels, 3x3 kernel
- Conv2: 32 → 64 channels, 3x3 kernel
- Conv3: 64 → 64 channels, 3x3 kernel
- Conv4: 64 → 128 channels, 3x3 kernel
- Fully Connected: 128 → 10 classes 