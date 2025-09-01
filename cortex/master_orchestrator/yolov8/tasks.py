from flask import current_app
import sys
from cortex.extensions import celery, socketio
from .utils import load_model, detect_using_yolov8


_yolov8_model = None


@celery.task(bind=True)
def detect(self, img_byts_file, modelinfo, device='cpu'):
    global _yolov8_model
    if _yolov8_model is None:
        _yolov8_model = load_model(weights=modelinfo['weights_path'], map_location=device)
    return detect_using_yolov8(_yolov8_model, img_byts_file, modelinfo, device)
