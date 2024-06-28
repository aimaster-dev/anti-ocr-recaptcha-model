import shortuuid
import torch
import torch.nn as nn
from torch.autograd import Variable
try:
    from .datasets import CaptchaData,CaptchaDataOne
    from .CONFIG import LettersInt
except:
    from datasets import CaptchaData,CaptchaDataOne
    from CONFIG import LettersInt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import time
from PIL import Image
import os, requests
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from MAIN import CrackLettesInt4

app = Flask(__name__)

def decode_base64_image(base64_string):
    if base64_string.startswith("data:image/png;base64,"):
        base64_string = base64_string.replace("data:image/png;base64,", "")
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image_path = 'temp.png'
    image.save(image_path)
    
    return image_path

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        image_file = request.files['image']
        image_path = 'temp.png'
        image_file.save(image_path)
    elif 'image_base64' in request.json:
        base64_string = request.json['image_base64']
        image_path = decode_base64_image(base64_string)
    else:
        return jsonify({"error": "No image provided."})

    try:
        pred_text = CrackLettesInt4().predict_one(image_path)
        return jsonify({'prediction': pred_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8000, debug=False)