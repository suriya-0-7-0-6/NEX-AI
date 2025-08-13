from cortex.extensions import celery, socketio
from .utils import add_arch_path_to_sys_path, load_model, detect_using_yolov7


_yolov7_model = None

@celery.task(bind=True)
def detect(self, img_byts_file, modelinfo, device='cpu'):
    add_arch_path_to_sys_path()
    global _yolov7_model
    if _yolov7_model is None:
        _yolov7_model = load_model(weights=modelinfo['weights_path'], map_location=device)
    return detect_using_yolov7(_yolov7_model, img_byts_file, modelinfo, device)
