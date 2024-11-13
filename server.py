from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_loss_data')
def get_loss_data():
    try:
        with open('training_logs.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'train_loss': [], 'val_loss': [], 'epochs': []})

if __name__ == '__main__':
    app.run(debug=True) 