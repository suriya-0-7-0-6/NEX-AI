TRAINING_CONFIGS = {
    "yolov8": [
        {"name": "epochs", "label": "Epochs", "type": "int", "default": 50, "min": 10, "max": 300},
        {"name": "imgsz", "label": "Image Size", "type": "int", "default": 640, "min": 240, "max": 2048},
        {"name": "batch_size", "label": "Batch Size", "type": "int", "default": 8, "min": 2, "max": 64},
        {"name": "dataset_yaml_path", "label": "Dataset Yaml Path", "type": "text"},
        {"name": "experiment_name", "label": "Experiment Name", "type": "text"}
    ],
    "yolov7": [
        {"name": "epochs", "label": "Epochs", "type": "int", "default": 50, "min": 10, "max": 300},
        {"name": "imgsz", "label": "Image Size", "type": "int", "default": 640, "min": 240, "max": 2048},
        {"name": "batch_size", "label": "Batch Size", "type": "int", "default": 8, "min": 2, "max": 64},
        {"name": "model_cfg_yaml_path", "label": "Model Config Yaml Path", "type": "text"},
        {"name": "dataset_yaml_path", "label": "Dataset Yaml Path", "type": "text"},
        {"name": "experiment_name", "label": "Experiment Name", "type": "text"},
        {"name": "weights", "label": "Weight Path", "type": "text"},
    ]    
}


INFERENCE_CONFIGS = {
    "pothole_detection": [
        {"name": "file_upload", "label": "File Upload", "type": "file", "accept": ".jpg, .jpeg, .png, .gif"}
    ],
    "sar_ship_detection": [
        {"name": "file_upload", "label": "File Upload", "type": "file", "accept": ".jpg, .jpeg, .png, .gif"}
    ],
    "signboard_detection": [
        {"name": "file_upload", "label": "File Upload", "type": "file", "accept": ".jpg, .jpeg, .png, .gif"}
    ]
}

PREPARE_DATASET_CONFIGS = {
    "yolov8": [
        {"name": "images_folder_path", "label": "Images Folder Path", "type": "text"},
        {"name": "via_annotation_file_path", "label": "Via Annotation File Path", "type": "text"}
    ]
}