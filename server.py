from flask import Flask, render_template, jsonify, request
import json
import os
import torch
from model import MNISTNet
from train import start_training
import threading
import logging
import datetime
from collections import deque
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

training_thread = None
training_status = {'status': 'idle'}
MODEL_HISTORY_FILE = 'model_history.json'
model_history = deque(maxlen=4)  # Keep only last 4 models

# Load existing model history if available
def load_model_history():
    try:
        with open(MODEL_HISTORY_FILE, 'r') as f:
            history = json.load(f)
            model_history.extend(history[-4:])  # Load only last 4 models
    except FileNotFoundError:
        pass

# Save model history
def save_model_history():
    with open(MODEL_HISTORY_FILE, 'w') as f:
        json.dump(list(model_history), f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training_route():
    global training_thread, training_status
    
    try:
        if training_thread and training_thread.is_alive():
            return jsonify({'error': 'Training already in progress'}), 400
        
        config = request.json
        logger.info(f"Received training config: {config}")
        
        # Convert string values to appropriate types
        config['batch_size'] = int(config['batch_size'])
        config['epochs'] = int(config['epochs'])
        config['num_filters'] = int(config['num_filters'])
        config['kernel_size'] = int(config['kernel_size'])
        config['learning_rate'] = float(config['learning_rate'])
        
        # Add to model history
        model_history.appendleft(config)
        save_model_history()
        
        training_status = {'status': 'running'}
        
        # Start training in a separate thread
        training_thread = threading.Thread(
            target=start_training,
            args=(config, training_status)
        )
        training_thread.start()
        
        logger.info("Training thread started successfully")
        return jsonify({'message': 'Training started'})
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_training_stats')
def get_training_stats():
    try:
        with open('training_logs.json', 'r') as f:
            data = json.load(f)
        data['status'] = training_status['status']
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({
            'train_loss': [], 
            'val_loss': [], 
            'train_acc': [],
            'val_acc': [],
            'epochs': [],
            'status': training_status['status']
        })
    except Exception as e:
        logger.error(f"Error getting training stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_thread, training_status
    if training_thread and training_thread.is_alive():
        training_status['status'] = 'completed'
        training_status['message'] = 'Training stopped by user'
        return jsonify({'message': 'Training stopped'})
    return jsonify({'message': 'No training in progress'}), 400

@app.route('/get_model_history')
def get_model_history():
    return jsonify(list(model_history))

if __name__ == '__main__':
    # Clear any existing training logs
    if os.path.exists('training_logs.json'):
        os.remove('training_logs.json')
    
    # Load existing model history
    load_model_history()
    
    app.run(debug=True, threaded=True) 