from flask import current_app
import sys
from cortex.extensions import celery, socketio
from .utils import load_model, detect_using_yolov8, train_using_yolov8
from .prepare_dataset import YOLOv8OBBProcessor


_yolov8_model = None


@celery.task(bind=True)
def detect(self, img_byts_file, modelinfo, device='cpu'):
    global _yolov8_model
    if _yolov8_model is None:
        _yolov8_model = load_model(weights=modelinfo['weights_path'], map_location=device)
    return detect_using_yolov8(_yolov8_model, img_byts_file, modelinfo, device)

@celery.task(bind=True)
def prepare_dataset(self, image_folder, json_path, output_folder, class_id):
    processor = YOLOv8OBBProcessor(image_folder, json_path, output_folder, class_id)
    processor.process()
    return f"Dataset prepared at {output_folder}"


@celery.task(bind=True)
def train(self, dataset_yaml_path, epochs, imgsz, batch_size, device, experiment_name):
    print(f"[YOLOv8 Bridge] Starting traininadsfasdg")
    return train_using_yolov8(dataset_yaml_path, epochs, imgsz, batch_size, device, experiment_name)