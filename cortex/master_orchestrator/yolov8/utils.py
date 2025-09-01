from flask import current_app, jsonify
from cortex.extensions import socketio
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
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

def load_model(weights, map_location='cpu'):
    model = YOLO(weights)
    return model

def detect_using_yolov8(model, img_byts_file, modelinfo, device):
    print(f"[YOLOv8 Bridge] Running inference")
    conf_thresh = modelinfo.get('confidence_threshold', 0.5)
    nms_thresh = modelinfo.get('nms_threshold', 0.45)
    input_size = modelinfo.get('input_size', 640)
    
    img_file = convert_img_file_to_numpy_array(img_byts_file)

    print("********************************************")
    print(model.info())
    print("********************************************")

    result_img_name = f"{modelinfo['id']}_result.png"
    result_img_file_path = os.path.join(current_app.config['RESULTS_DIR'], result_img_name)

    result_url = f'/uploads/{result_img_name}'
    socketio.emit(
        'result',
        {'result_img_file_path': result_img_file_path, 'result_url': result_url}
    )

    return 
