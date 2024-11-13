from flask import Flask, render_template, jsonify, request
import json
import os
import threading
from datetime import datetime
from state import model_plots, training_status, stop_flag
import torch
from torchvision import datasets, transforms
import random
import base64
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

def get_model_plots():
    if os.path.exists('training_logs.json'):
        with open('training_logs.json', 'r') as f:
            training_logs = json.load(f)
        return training_logs
    return None

@app.route('/')
def index():
    plots_data = get_model_plots()
    return render_template('index.html', plots_data=plots_data)

@app.route('/get_model_history')
def get_model_history():
    try:
        if os.path.exists('training_logs.json'):
            with open('training_logs.json', 'r') as f:
                logs = json.load(f)
                # Convert logs to list of models with their configs
                history = []
                for model_name, data in logs.items():
                    if 'config' in data:
                        config = data['config']
                        config['model_name'] = model_name
                        config['timestamp'] = datetime.now().isoformat()
                        history.append(config)
                return jsonify(history)
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_training_stats')
def get_training_stats():
    try:
        current_model = None
        models_data = []
        
        if os.path.exists('training_logs.json'):
            with open('training_logs.json', 'r') as f:
                logs = json.load(f)
                for model_name, data in logs.items():
                    models_data.append([model_name, data])
                    if data.get('status') == 'running':
                        current_model = model_name
        
        return jsonify({
            'status': training_status.get('status', 'idle'),
            'error': training_status.get('error', None),
            'current_model': current_model,
            'models': models_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    global stop_flag
    
    try:
        config = request.json
        model_name = config.get('model_name', 'default_model')
        
        # Reset training status
        training_status['status'] = 'running'
        
        # Import here to avoid circular import
        from train import start_training
        
        # Start training in a separate thread
        stop_flag = threading.Event()
        training_thread = threading.Thread(
            target=start_training,
            args=(config, training_status, model_name)
        )
        training_thread.start()
        
        return jsonify({'status': 'Training started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_training', methods=['POST'])
def stop_training():
    if stop_flag:
        stop_flag.set()
        return jsonify({'status': 'Training stop signal sent'})
    return jsonify({'status': 'No training in progress'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Clear the model_plots dictionary
        model_plots.clear()
        
        # Delete or clear the training_logs.json file
        if os.path.exists('training_logs.json'):
            os.remove('training_logs.json')
            
        return jsonify({'status': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_predictions/<model_name>')
def get_predictions(model_name):
    try:
        print(f"Getting predictions for model: {model_name}")
        
        # Set matplotlib to use non-GUI backend
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Check if model file exists
        model_path = f'models/{model_name}.pth'
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return jsonify({
                'success': False,
                'error': f'Model file not found: {model_path}'
            })
        
        # Check if training logs exist
        if not os.path.exists('training_logs.json'):
            return jsonify({
                'success': False,
                'error': 'Training logs not found'
            })
            
        # Load test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Get model config
        with open('training_logs.json', 'r') as f:
            logs = json.load(f)
            if model_name not in logs:
                return jsonify({
                    'success': False,
                    'error': f'Model {model_name} not found in training logs'
                })
            config = logs[model_name]['config']
        
        # Initialize model
        from model import MNISTNet
        model = MNISTNet(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Randomly select 10 images
        indices = random.sample(range(len(test_dataset)), 10)
        images = []
        labels = []
        predictions = []
        
        # Get predictions
        for idx in indices:
            img, label = test_dataset[idx]
            img = img.to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(img.unsqueeze(0))
                pred = output.argmax(dim=1).item()
            
            # Convert tensor to image for display
            img_display = img.cpu().squeeze().numpy()
            
            # Create figure for this image
            fig = plt.figure(figsize=(2, 2), dpi=100)
            plt.imshow(img_display, cmap='gray')
            plt.axis('off')
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            images.append(img_str)
            labels.append(int(label))
            predictions.append(pred)
        
        print(f"Successfully generated predictions for {len(predictions)} images")
        
        return jsonify({
            'success': True,
            'images': images,
            'labels': labels,
            'predictions': predictions
        })
            
    except Exception as e:
        print(f"Error in get_predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 