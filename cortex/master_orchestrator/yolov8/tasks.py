from flask import current_app
import sys
import os
import torch
from cortex.extensions import celery, socketio
from .utils import load_model, detect_using_yolov8, train_using_yolov8, bulk_detect_using_yolov8
from .prepare_dataset import YOLOv8OBBProcessor


_yolov8_model = None

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

    print(f"[set_active_arch] Now using only {arch_name}")
    print("sys.path =", sys.path)

@celery.task(bind=True)
def detect(self, img_byts_file, modelinfo, output_folder_path):
    set_active_arch(modelinfo['dnnarch'])
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global _yolov8_model
    if _yolov8_model is None:
        _yolov8_model = load_model(weights=modelinfo['weights_path'], map_location=map_location)
    return detect_using_yolov8(_yolov8_model, img_byts_file, modelinfo, map_location, output_folder_path)

@celery.task(bind=True)
def prepare_dataset(self, params):
    set_active_arch(params['dnnarch'])
    processor = YOLOv8OBBProcessor(params['images_folder_path'], params['via_annotation_file_path'], params['output_folder_path'])
    processor.process()
    return f"Dataset prepared at {params['output_folder_path']}"

@celery.task(bind=True)
def train(self, params):
    set_active_arch(params['dnnarch'])
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_using_yolov8(map_location, params)


def bulk_inference(image_file, modelinfo):
    set_active_arch(modelinfo['dnnarch'])
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global _yolov8_model
    if _yolov8_model is None:
        _yolov8_model = load_model(weights=modelinfo['weights_path'], map_location=map_location)
    return bulk_detect_using_yolov8(_yolov8_model, image_file, modelinfo, map_location)