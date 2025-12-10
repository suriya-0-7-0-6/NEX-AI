import os
import json
import math
import cv2
import numpy as np
import random
import shutil
import yaml
from flask import current_app
from cortex.extensions import socketio

class YOLOv8OBBProcessor:
    def __init__(self, image_folder, json_path, output_folder, class_id=0):
        self.image_folder = image_folder
        self.json_path = json_path
        self.output_folder = output_folder
        self.class_id = class_id

    def rectangle_coordinates(self, x1, y1, width, height, alpha_deg, beta_deg):
        alpha = math.radians(alpha_deg)
        beta = math.radians(beta_deg)

        x2 = x1 + height * math.cos(alpha)
        y2 = y1 + height * math.sin(alpha)
        x3 = x2 + width * math.cos(beta)
        y3 = y2 + width * math.sin(beta)
        x4 = x1 + width * math.cos(beta)
        y4 = y1 + width * math.sin(beta)

        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    def draw_regions_on_image(self, img, regions):
        all_polygons = []

        for region in regions:
            shape = region["shape_attributes"]
            x1, y1 = shape["x"], shape["y"]
            width, height = shape["width"], shape["height"]
            alpha, beta = shape["alpha"], shape["beta"]

            corners = self.rectangle_coordinates(x1, y1, width, height, alpha, beta)
            pts = [(int(x), int(y)) for (x, y) in corners]

            cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

            for (x, y) in pts:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

            all_polygons.append(corners)

        return img, all_polygons

    def save_yolo_obb_annotations(self, txt_path, polygons, img_w, img_h):
        with open(txt_path, "w") as f:
            for poly in polygons:
                norm_coords = []
                for (x, y) in poly:
                    norm_coords.append(x / img_w)
                    norm_coords.append(y / img_h)

                coords_str = " ".join([f"{c:.6f}" for c in norm_coords])
                f.write(f"{self.class_id} {coords_str}\n")

    def split_and_save_dataset(self, label_files):
        random.shuffle(label_files)
        n = len(label_files)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        splits = {
            "train": label_files[:n_train],
            "val": label_files[n_train:n_train + n_val],
            "test": label_files[n_train + n_val:]
        }

        for split, files in splits.items():
            split_img_dir = os.path.join(self.output_folder, "images", split)
            split_lbl_dir = os.path.join(self.output_folder, "labels", split)
            os.makedirs(split_img_dir, exist_ok=True)
            os.makedirs(split_lbl_dir, exist_ok=True)

            for img_path, lbl_path in files:
                shutil.copy(img_path, os.path.join(split_img_dir, os.path.basename(img_path)))
                shutil.copy(lbl_path, os.path.join(split_lbl_dir, os.path.basename(lbl_path)))

            print(f"[SPLIT] {split}: {len(files)} samples")
        socketio.emit(
            'live_logs',
            {
                'progress': {
                    "level": "info",
                    'status': f'Dataset split into train, test, val', 
                    'logs': {
                        'length of train set': len(splits['train']),
                        'length of val set': len(splits['val']),
                        'length of test set': len(splits['test'])
                    }
                }
            }
        )

    def save_dataset_yaml(self):
        yaml_path = os.path.join(self.output_folder, "dataset.yaml")
        yaml_content = {
            "path": ".",
            "train": os.path.join(self.output_folder, "images/train"),
            "val": os.path.join(self.output_folder, "images/val"),
            "test": os.path.join(self.output_folder, "images/test"),
            "nc": 1,
            "names": ["ship"]
        }

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

    def process(self):
        print(f"[PROCESS] Loading annotations from: {self.json_path}")
        socketio.emit(
            'live_logs',
            {
                'progress': {
                    "level": "info",
                    'status': f'Loading annotations from: {self.json_path}', 
                    'logs': {}
                }
            }
        )

        with open(self.json_path, "r") as f:
            annotations = json.load(f)

        os.makedirs(self.output_folder, exist_ok=True)

        label_files = []

        for img_name, ann in annotations.items():
            img_path = os.path.join(self.image_folder, img_name)
            if not os.path.exists(img_path):
                socketio.emit(
                    'live_logs',
                    {
                        'progress': {
                            "level": "warning",
                            'status': f'Image not found: {img_path}', 
                            'logs': {}
                        }
                    }
                )
                continue

            img = cv2.imread(img_path)
            if img is None:
                socketio.emit(
                    'live_logs',
                    {
                        'progress': {
                            "level": "warning",
                            'status': f'Could not read image: {img_path}', 
                            'logs': {}
                        }
                    }
                )
                continue

            img_h, img_w = img.shape[:2]
            regions = ann.get("regions", [])
            img_out, polygons = self.draw_regions_on_image(img, regions)

            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(self.output_folder, txt_name)
            self.save_yolo_obb_annotations(txt_path, polygons, img_w, img_h)
            print(f"[SAVED YOLO OBB] {txt_path}")

            label_files.append((img_path, txt_path))

        self.split_and_save_dataset(label_files)
        self.save_dataset_yaml()
        socketio.emit(
            'live_logs',
            {
                'progress': {
                    "level": "info",
                    'status': f'Dataset prepared successfully!', 
                    'logs': {
                        'dataset_folder_path': self.output_folder
                    }
                }
            }
        )
