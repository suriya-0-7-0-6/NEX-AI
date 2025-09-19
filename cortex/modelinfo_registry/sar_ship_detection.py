import os
from cortex.app_configurations import BaseConfigurations


CONFIG = {
  "id": "sar_ship_detection",
  "task_name": "sar_ship_detection",
  "description": "Detect ships in SAR images.",
  "dnnarch": "yolov8",
  "dnnarch_repo_path": None,
  "weights_path": os.path.join(BaseConfigurations.WEIGHTS_DIR, 'yolov8_sar_ship.pt'),
  "input_size": [640, 640],
  "classes": [
    "ship"
  ],
  "confidence_threshold": 0.1,
  "nms_threshold": 0.45
}