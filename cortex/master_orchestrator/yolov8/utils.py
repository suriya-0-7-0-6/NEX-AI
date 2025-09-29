from flask import current_app, jsonify
from cortex.extensions import socketio
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np
import json
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

def detect_using_yolov8(model, img_byts_file, modelinfo, device, output_folder_path):
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
        project=output_folder_path, 
        name=modelinfo['id'] + modelinfo['dnnarch'],
        exist_ok=True
    )

    predicted_img_array = results[0].plot() 
    result_img_name = f"{modelinfo['id']}_{modelinfo['dnnarch']}_{time.time()}.png"
    result_img_file_path = f"{output_folder_path}/{result_img_name}"
    cv2.imwrite(result_img_file_path, predicted_img_array)

    result_url = f'/upload_single_inference_image/{result_img_name}'

    socketio.emit(
        'Single_inference_result',
        {
            'progress': {
                'status': 'Inference completed successfully!', 
                'result': {
                    'result_img_file_path': result_img_file_path, 
                    'result_url': result_url
                }
            }
        }
    )
    return



def bulk_detect_using_yolov8(model, image_folder, output_folder, modelinfo, map_location):
    os.makedirs(output_folder, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in image_extensions]

    detection_results = {}

    for image_file in image_files:
        file_path = os.path.join(image_folder, image_file)
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        np_img = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            continue  

        results = model.predict(
            source=img,
            imgsz=modelinfo.get("input_size", 640),
            conf=modelinfo.get("confidence_threshold", 0.25),
            save=False,
            device=map_location
        )

        filename = secure_filename(image_file)
        output_json = {
            filename: {
                "filename": filename,
                "size": "-1",
                "regions": [],
                "file_attributes": {}
            }
        }

        if results and hasattr(results[0], 'obb') and results[0].obb is not None:
            obb_data = results[0].obb
            xyxyxyxy = obb_data.xyxyxyxy.cpu().numpy()
            confs = obb_data.conf.cpu().numpy()
            clss = obb_data.cls.cpu().numpy()
            class_names = results[0].names

            for i, pts in enumerate(xyxyxyxy):
                pts = np.array(pts).flatten()
                if pts.shape[0] != 8:
                    continue

                all_points_x = [int(round(pts[j])) for j in range(0, 8, 2)]
                all_points_y = [int(round(pts[j])) for j in range(1, 8, 2)]

                class_id = int(clss[i])
                label = class_names[class_id]

                region = {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": all_points_x,
                        "all_points_y": all_points_y
                    },
                    "region_attributes": {
                        "Label": label
                    }
                }
                output_json[filename]["regions"].append(region)

        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, "w") as f:
            json.dump(output_json, f, indent=4)

        detection_results[filename] = output_json[filename]

    socketio.emit(
        'bulk_inference_result',
        {
            'progress': {
                'status': 'Bulk inference completed successfully!', 
                'result': {
                    'result_dir': output_folder, 
                }
            }
        }
    )



def train_using_yolov8(data, epochs, imgsz, batch_size, device, name, output_folder_path):
    print(f"[YOLOv8 Bridge] Starting training")
    
    socketio.emit(
        'train_progress',
        {
            'progress': {
                'status': 'started',
                'result': {
                    'data': data,
                    'epochs': epochs,
                    'imgsz': imgsz,
                    'batch': batch_size,
                    'device': device,
                    'name': name
                }
            }
        }
    )

    model = YOLO("yolov8n-obb.pt")

    name=f"exp_{name}_{int(time.time())}"

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=output_folder_path,
        name=name,
        workers=0,
        exist_ok=True
    )

    result_dir = os.path.join(output_folder_path, name)

    socketio.emit(
        'train_results',
        {
            'progress': {
                'status': 'Training completed successfully!', 
                'result': {
                    'result_dir': result_dir, 
                }
            }
        }
    )

