from flask import Blueprint, send_from_directory, current_app

files_api = Blueprint('file_manager_routes', __name__)

@files_api.route('/uploads/<filename>', methods=['GET'])
def fetch_and_serve_requested_file(filename):
    return send_from_directory(current_app.config['RESULTS_DIR'], filename)