import os
import requests

API_URL = "http://localhost:5000/has_detection"
PROBLEM_ID = "pothole_detection"
IMG_DIR = "/path/to/images"
YES_DIR = "/path/to/with_detections"
NO_DIR = "/path/to/no_detections"

os.makedirs(YES_DIR, exist_ok=True)
os.makedirs(NO_DIR, exist_ok=True)

for img_name in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_name)
    if not img_path.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    with open(img_path, "rb") as f:
        resp = requests.post(API_URL, data={"problem_id": PROBLEM_ID}, files={"upload_image": f})
        result = resp.json()["result"]

        if resp.status_code == 200:
            response_data = resp.json()
            result = response_data.get("has_detection", False)
        else:
            print(f"Failed to fetch data for {img_name}, Status Code: {resp.status_code}")
            result = False 
    
    if result:
        os.rename(img_path, os.path.join(YES_DIR, img_name))
    else:
        os.rename(img_path, os.path.join(NO_DIR, img_name))

    print(f"{img_name}: {result}")
