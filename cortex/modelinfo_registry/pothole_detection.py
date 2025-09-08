import os
from cortex.app_configurations import BaseConfigurations


CONFIG = {
  "id": "pothole_detection",
  "task_name": "pothole_detection",
  "description": "Detect potholes in images.",
  "dnnarch": "yolov5",
  "dnnarch_repo_path": os.path.join(BaseConfigurations.MODEL_ARCHS_DIR, 'yolov5'),
  "weights_path": os.path.join(BaseConfigurations.WEIGHTS_DIR, 'yolov5_pothole.pt'),
  "input_size": [640, 640],
  "classes": [
    "pothole",
    "big_pothole",
    "water_pothole"
  ],
  "confidence_threshold": 0.5,
  "nms_threshold": 0.45
}


