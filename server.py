from flask import Flask, render_template, jsonify, request
import json
import os
import threading
from datetime import datetime
from state import model_plots, training_status, stop_flag

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

if __name__ == '__main__':
    app.run(debug=True) 