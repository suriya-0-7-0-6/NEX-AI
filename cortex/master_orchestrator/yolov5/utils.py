from flask import current_app, jsonify
from cortex.extensions import socketio
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
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

def prepare_input(img, input_size, device='cpu'):
    from utils.augmentations import letterbox
    print(f"[YOLOv5 Bridge] Preparing input image with size: {input_size}")
    img_resized = letterbox(img, input_size, stride=32, auto=True)[0]
    img_resized = img_resized.transpose((2, 0, 1))[::-1]
    img_resized = np.ascontiguousarray(img_resized)
    model_input_img = torch.from_numpy(img_resized).to(device).float() / 255.0
    return model_input_img.unsqueeze(0)

def draw_detections(img, detections, classes, conf_thresh):
    drwn_img = Image.fromarray(img)
    draw = ImageDraw.Draw(drwn_img)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf > conf_thresh:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 20), f"{classes[int(cls)]}: {conf:.2f}", fill="red")
    return drwn_img

def load_model(weights, map_location='cpu'):
    print(f" YOLOV5: {sys.path}")
    from models.common import DetectMultiBackend
    model = DetectMultiBackend(weights, device=map_location)
    return model

def detect_using_yolov5(model, img_byts_file, modelinfo, device):
    from utils.torch_utils import select_device
    from utils.general import non_max_suppression, scale_boxes

    

    conf_thresh = modelinfo.get('confidence_threshold', 0.5)
    nms_thresh = modelinfo.get('nms_threshold', 0.45)
    input_size = modelinfo.get('input_size', 640)

    img_file = convert_img_file_to_numpy_array(img_byts_file)
    img_tensor = prepare_input(img_file, input_size, device)
    original_shape = img_file.shape[:2]

    with torch.no_grad():
        pred = model(img_tensor)
        detections = non_max_suppression(pred, conf_thresh, nms_thresh)[0]

    output = []
    if detections is not None and len(detections):
        detections[:, :4] = scale_boxes(img_tensor.shape[2:], detections[:, :4], original_shape).round()

        for box in detections:
            x1, y1, x2, y2, conf, cls_id = box
            if conf < conf_thresh:
                continue
            label = modelinfo['classes'][int(cls_id)]
            output.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': label
            })

        drwn_img = draw_detections(img_file, detections, modelinfo['classes'], conf_thresh)
    else:
        drwn_img = Image.fromarray(img_file)

    result_img_name = f"{modelinfo['id']}_result.png"
    result_img_file_path = os.path.join(current_app.config['RESULTS_DIR'], result_img_name)
    drwn_img.save(result_img_file_path)

    # Emit results
    result_url = f'/uploads/{result_img_name}'
    socketio.emit(
        'result',
        {'result_img_file_path': result_img_file_path, 'result_url': result_url, 'detections': output}
    )

    return result_img_file_path