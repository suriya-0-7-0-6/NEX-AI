from flask import current_app, jsonify
from cortex.extensions import socketio
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
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

def rescale_bounding_boxes(old_size, new_size, boxes):
  old_height, old_width = old_size
  new_height, new_width = new_size
  scale_x = new_width / old_width
  scale_y = new_height / old_height
  boxes_rescaled = boxes.clone()
  boxes_rescaled[:, 0] *= scale_x  
  boxes_rescaled[:, 1] *= scale_y 
  boxes_rescaled[:, 2] *= scale_x
  boxes_rescaled[:, 3] *= scale_y
  return boxes_rescaled.round()

def draw_detections(img, detections, classes, conf_thresh):
    drwn_img = Image.fromarray(img)
    draw = ImageDraw.Draw(drwn_img)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf > conf_thresh:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-20), f"{classes[int(cls)]}: {conf:.2f}", fill="red")
    return drwn_img

def load_model(weights, map_location='cpu'):
    print(f" YOLOV7: {sys.path}")
    from models.experimental import attempt_custom_load
    return attempt_custom_load(weights=weights, map_location=map_location)

def perform_detection(model, img_tensor, conf_thresh, nms_thresh):
    from utils.general import non_max_suppression
    with torch.no_grad():
        detections = model(img_tensor)[0]
        detections = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
    return detections

def parse_detections(detections, modelinfo, img_tensor, original_shape, conf_thresh):
    output = []
    detections[:, :4] = rescale_bounding_boxes((img_tensor.shape[2],img_tensor.shape[3]), original_shape, detections[:, :4])
    for box in detections if detections is not None else []:
        x1, y1, x2, y2, conf, cls_id = box
        if conf < conf_thresh:
            continue
        label = modelinfo['classes'][int(cls_id)]
        output.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(conf),
            'class': label
        })
    return output

def detect_using_yolov7(model, img_byts_file, modelinfo, device='cpu'):
    print(f"[YOLOv7 Bridge] Running inference")
    conf_thresh = modelinfo.get('confidence_threshold', 0.5)
    nms_thresh = modelinfo.get('nms_threshold', 0.45)
    input_size = modelinfo.get('input_size', 640)

    img_file, img_tensor, original_shape = prepare_input(img_byts_file, input_size, device)
    detections = perform_detection(model, img_tensor, conf_thresh, nms_thresh)

    output = []
    if detections is not None and len(detections) > 0:
        output = parse_detections(detections, modelinfo, img_tensor, original_shape, conf_thresh)
        drwn_img = draw_detections(img_file, detections, modelinfo['classes'], conf_thresh)
    else:
        drwn_img = Image.fromarray(img_file)
   
    result_img_name = f"{modelinfo['id']}_{modelinfo['dnnarch']}_{time.time()}.png"
    result_img_file_path = f"{current_app.config['RESULTS_DIR']}/{result_img_name}"
    drwn_img.save(result_img_file_path)

    result_url = f'/uploads/{result_img_name}'
    socketio.emit(
        'result',
        {'result_img_file_path': result_img_file_path, 'result_url': result_url, 'detections': output if detections is not None else []}
    )
    return result_img_file_path
