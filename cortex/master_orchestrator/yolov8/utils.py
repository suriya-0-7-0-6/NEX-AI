from flask import current_app, jsonify
from cortex.extensions import socketio
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO
import numpy as np
import time
import torch
import cv2
import sys
import os

def convert_img_file_to_numpy_array(file_bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    cv2_img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if cv2_img_bgr is None:
        return jsonify({'error': 'Invalid image format'}), 400
    cv2_img_rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)
    img = np.array(cv2_img_rgb)
    return img

def prepare_input(file_bytes, input_size, device='cpu'):
    img = convert_img_file_to_numpy_array(file_bytes)
    original_shape = img.shape[:2]
    model_input_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size[0], input_size[1])),
        transforms.ToTensor(),
    ])(img).unsqueeze(0).to(device).float()
    return img, model_input_img, original_shape

def load_model(weights, map_location='cpu'):
    model = YOLO(weights)
    return model

def detect_using_yolov8(model, img_byts_file, modelinfo, device):
    print(f"[YOLOv8 Bridge] Running inference")
    conf_thresh = modelinfo.get('confidence_threshold', 0.5)
    nms_thresh = modelinfo.get('nms_threshold', 0.45)
    input_size = modelinfo.get('input_size', 640)
    
    img_file, img_tensor, original_shape = prepare_input(img_byts_file, input_size, device)

    results = model.predict(
        source=img_tensor,
        save=False,     
        imgsz=640,
        conf=0.25,
        project=current_app.config['RESULTS_DIR'],
        name=modelinfo['id'] + modelinfo['dnnarch'],
        exist_ok=True
    )

    predicted_img_array = results[0].plot() 
    result_img_name = f"{modelinfo['id']}_{modelinfo['dnnarch']}_{time.time()}.png"
    result_img_file_path = f"{current_app.config['RESULTS_DIR']}/{result_img_name}"
    cv2.imwrite(result_img_file_path, predicted_img_array)

    result_url = f'/uploads/{result_img_name}'
    socketio.emit(
        'result',
        {'result_img_file_path': result_img_file_path, 'result_url': result_url}
    )

    return 
