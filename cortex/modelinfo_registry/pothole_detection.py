import os
from cortex.app_configurations import BaseConfigurations


CONFIG = {
  "id": "pothole_detection",
  "task_name": "pothole_detection",
  "description": "Detect potholes in images.",
  "dnnarch": "yolov8",
  "dnnarch_repo_path": None,
  "weights_path": os.path.join(BaseConfigurations.WEIGHTS_DIR, 'yolov8_pothole.pt'),
  "input_size": [640, 640],
  "classes": [
    "pothole",
    "big_pothole",
    "water_pothole"
  ],
  "confidence_threshold": 0.5,
  "nms_threshold": 0.45
}


