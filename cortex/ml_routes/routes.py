from flask import Blueprint, request, current_app, render_template
import importlib

from cortex.master_orchestrator import fetch_all_problem_ids, fetch_configs
from cortex.forms import AiInferenceForm

ml_api = Blueprint('ml_routes', __name__)

@ml_api.route('/get_all_problem_ids', methods=['GET'])
def get_all_problem_ids():

    all_problem_ids = fetch_all_problem_ids()

    return {'problem_ids': all_problem_ids}, 200



@ml_api.route('/get_problem_config/<problem_id>', methods=['GET'])
def get_problem_config(problem_id):
    configs = fetch_configs(problem_id)

    if not configs:
        return {'error': 'Configuration file is empty'}, 404

    return {'configs': configs}, 200



@ml_api.route('/ai_inference', methods=['GET', 'POST'])
def ai_inference():
    AIForm = AiInferenceForm()
    AIForm.problem_id.choices = fetch_all_problem_ids()

    if request.method == 'POST':
        if AIForm.validate_on_submit():
            info = "Form submitted successfully"
            problem_id = AIForm.problem_id.data
            upload_image_bytes_file = AIForm.upload_image.data.read()

            if not upload_image_bytes_file:
                return {'error': 'No file uploaded'}, 400
            
            if not problem_id:
                return {'error': 'Problem ID is required'}, 400

            modelinfo = fetch_configs(problem_id)
            
            if not modelinfo:
                return {'error': 'Model configuration not found'}, 404

            task = importlib.import_module(f"cortex.master_orchestrator.yolov7")

            task.detect.apply_async(
                args=[upload_image_bytes_file, modelinfo, 'cpu']
            )

            return render_template('ai_pages/ai_inference.html', form=AIForm, info=info), 200

    return render_template('ai_pages/ai_inference.html', form=AIForm, info=""), 200
    