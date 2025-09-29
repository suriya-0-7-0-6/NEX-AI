from flask import Blueprint, send_from_directory, current_app
import os

files_api = Blueprint('file_manager_routes', __name__)

@files_api.route('/uploads/<filename>', methods=['GET'])
def fetch_and_serve_requested_file(filename):
    return send_from_directory(current_app.config['RESULTS_DIR'], filename)

@files_api.route('/upload_single_inference_image/<filename>', methods=['GET'])
def upload_single_inference_image(filename):
    return send_from_directory(os.path.join(current_app.config['LOGS_DIR'], "single_image_inferences"), filename)

