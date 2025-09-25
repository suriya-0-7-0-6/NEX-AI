import os
import requests
import json
import argparse

def send_image_for_detection(image_path, problem_id, endpoint_url):
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {'problem_id': problem_id}

        try:
            response = requests.post(endpoint_url, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed on {image_path}: {e}")
            return None

def save_detection_json(response_json, output_dir, image_filename):
    output_filename = os.path.splitext(image_filename)[0] + ".json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(response_json, f, indent=4)
    print(f"[SAVED] {output_path}")

def main(image_dir, output_dir, problem_id, endpoint_url):
    if not os.path.isdir(image_dir):
        print(f"[ERROR] Image directory not found: {image_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print("[INFO] No valid image files found in the directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"[INFO] Processing {image_path}...")

        result = send_image_for_detection(image_path, problem_id, endpoint_url)
        if result:
            save_detection_json(result, output_dir, image_file)

if __name__ == "__main__":
   
    image_dir = "grey_matter/datasets/yolov8/images/test"
    output_dir = "/home/jarvis/Desktop/projects/stash/output"
    problem_id = "sar_ship_detection"
    url = "http://localhost:5000/get_detections"

    main(image_dir, output_dir, problem_id, url)
