from flask import current_app
import sys
from cortex.extensions import celery, socketio
from .utils import load_model, detect_using_yolov5


_yolov5_model = None

def add_arch_path_to_sys_path():
    import site
    YOLOV5_DIR = current_app.config['MODEL_ARCHS_DIR'] + '/yolov5'
    site.addsitedir(YOLOV5_DIR)


@celery.task(bind=True)
def detect(self, img_byts_file, modelinfo, device='cpu'):
    add_arch_path_to_sys_path()
    global _yolov5_model
    if _yolov5_model is None:
        _yolov5_model = load_model(weights=modelinfo['weights_path'], map_location=device)
    return detect_using_yolov5(_yolov5_model, img_byts_file, modelinfo, device)
