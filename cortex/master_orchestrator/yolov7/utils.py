from flask import current_app, jsonify
from cortex.extensions import socketio
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import numpy as np
import torch
import cv2
import sys
import os


def add_arch_path_to_sys_path():
    YOLOV7_DIR = current_app.config['MODEL_ARCHS_DIR'] + '/yolov7'
    if YOLOV7_DIR not in sys.path:
        sys.path.insert(0, YOLOV7_DIR)

def convert_img_file_to_numpy_array(file_bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    cv2_img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if cv2_img_bgr is None:
        return jsonify({'error': 'Invalid image format'}), 400
    cv2_img_rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)
    img = np.array(cv2_img_rgb)
    return img

def prepare_input(img, input_size, device='cpu'):
    print(f"[YOLOv7 Bridge] Preparing input image with size: {input_size}")
    model_input_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size[0], input_size[1])),
        transforms.ToTensor(),
    ])(img).unsqueeze(0).to(device).float()
    return model_input_img

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
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=15)  
    for det in detections[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf > conf_thresh:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-20), f"{classes[int(cls)]}: {conf:.2f}", fill="red", font=font)
    return drwn_img

def load_model(weights, map_location='cpu'):
    from grey_matter.model_archs.yolov7.models.experimental import attempt_custom_load
    return attempt_custom_load(weights=weights, map_location=map_location)

def detect_using_yolov7(model, img_byts_file, modelinfo, device='cpu'):
    from grey_matter.model_archs.yolov7.utils.general import non_max_suppression
    print(f"[YOLOv7 Bridge] Running inference")
    conf_thresh = modelinfo.get('confidence_threshold', 0.5)
    nms_thresh = modelinfo.get('nms_threshold', 0.45)
    input_size = modelinfo.get('input_size', 640)

    img_file = convert_img_file_to_numpy_array(img_byts_file)

    model_input_img = prepare_input(img_file, input_size, device)

    with torch.no_grad():
        detections = model(model_input_img)[0]
        detections = non_max_suppression(detections, conf_thresh, nms_thresh)

    if detections[0] is not None:
        detections[0][:, :4] = rescale_bounding_boxes((model_input_img.shape[2],model_input_img.shape[3]), (img_file.shape[0], img_file.shape[1]), detections[0][:, :4])

    output = []
    for box in detections[0] if detections is not None else []:
        x1, y1, x2, y2, conf, cls_id = box
        if conf < modelinfo.get("confidence_threshold", 0.25):
            continue
        label = modelinfo['classes'][int(cls_id)]
        output.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(conf),
            'class': label
        })
    
    drwn_img = draw_detections(img_file, detections, modelinfo['classes'], conf_thresh)
    result_img_name = f"{modelinfo['id']}_result.png"

    result_img_file_path = os.path.join(current_app.config['RESULTS_DIR'], result_img_name)
    drwn_img.save(result_img_file_path)

    result_url = f'/uploads/{result_img_name}'
    socketio.emit(
        'result',
        {'result_img_file_path': result_img_file_path, 'result_url': result_url, 'detections': output}
    )

    return result_img_file_path
