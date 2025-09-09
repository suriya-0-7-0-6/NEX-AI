from flask import current_app
import torch
import sys
import os
from cortex.extensions import celery, socketio
from .utils import load_model, detect_using_yolov5


_yolov5_model = None

def set_active_arch(arch_name):
    base_dir = current_app.config['MODEL_ARCHS_DIR']
    arch_dir = os.path.join(base_dir, arch_name)

    sys.path = [p for p in sys.path if not p.startswith(base_dir)]
    sys.path.insert(0, arch_dir)

    to_remove = []
    for mod_name, mod in list(sys.modules.items()):
        try:
            mod_file = getattr(mod, "__file__", "")
        except Exception:
            continue
        if mod_file and mod_file.startswith(base_dir) and arch_name not in mod_file:
            to_remove.append(mod_name)

    for mod_name in to_remove:
        sys.modules.pop(mod_name, None)


@celery.task(bind=True)
def detect(self, img_byts_file, modelinfo):
    global _yolov5_model
    set_active_arch(modelinfo['dnnarch'])
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"map_location: {map_location}")
    # map_location = torch.device("cpu")
    if _yolov5_model is None:
        _yolov5_model = load_model(weights=modelinfo['weights_path'], map_location=map_location)
    return detect_using_yolov5(_yolov5_model, img_byts_file, modelinfo, map_location)