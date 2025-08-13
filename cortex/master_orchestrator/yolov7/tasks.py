from cortex.extensions import celery, socketio
from utils import attempt_load, detect_using_yolov7

_yolov7_model = None

@celery.task(bind=True)
def detect(self, img_byts_file, modelinfo, device='cpu'):
    global _yolov7_model
    if _yolov7_model is None:
        _yolov7_model = attempt_load(weights=modelinfo['weights_path'], map_location=device)
    return detect_using_yolov7(_yolov7_model, img_byts_file, modelinfo, device)
