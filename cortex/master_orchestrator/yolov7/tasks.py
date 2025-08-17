from flask import current_app
import sys
from cortex.extensions import celery, socketio
from .utils import load_model, detect_using_yolov7


_yolov7_model = None

def add_arch_path_to_sys_path():
    import site
    YOLOV7_DIR = current_app.config['MODEL_ARCHS_DIR'] + '/yolov7'
    site.addsitedir(YOLOV7_DIR)


@celery.task(bind=True)
def detect(self, img_byts_file, modelinfo, device='cpu'):
    add_arch_path_to_sys_path()
    global _yolov7_model
    if _yolov7_model is None:
        _yolov7_model = load_model(weights=modelinfo['weights_path'], map_location=device)
    return detect_using_yolov7(_yolov7_model, img_byts_file, modelinfo, device)
