from flask import Blueprint, request, current_app, render_template
import importlib

from cortex.master_orchestrator import fetch_all_problem_ids, fetch_configs, fetch_all_model_archs
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
    AIForm.models_list.choices = fetch_all_model_archs()

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
            
            try:
                dnnarch = modelinfo['dnnarch']
                tasks_module = importlib.import_module(f"cortex.master_orchestrator.{dnnarch}.tasks")
                detect_task = getattr(tasks_module, 'detect')
            except (ImportError, AttributeError) as e:
                return {'error': f"Failed to load detect task: {e}"}, 500

            detect_task.apply_async(args=[upload_image_bytes_file, modelinfo])

            return render_template('ai_pages/ai_inference.html', form=AIForm, info=info), 200

    return render_template('ai_pages/ai_inference.html', form=AIForm, info=""), 200
    


import cortex.master_orchestrator.yolov5.tasks as yolov5_tasks
from cortex.master_orchestrator.yolov5.utils import check_if_any_detection_present

@ml_api.route('/has_detection', methods=['POST'])
def has_detection():
    import torch
    import os
    problem_id = request.form.get("problem_id")
    file = request.files.get("image")

    if not problem_id:
        return {"error": "Problem ID is required"}, 400
    if not file:
        return {"error": "No image uploaded"}, 400  

    modelinfo = fetch_configs(problem_id)
    if not modelinfo:
        return {"error": "Model configuration not found"}, 404

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolov5_tasks.set_active_arch(modelinfo['dnnarch'])
    map_location = torch.device("cpu")
    
    if yolov5_tasks._yolov5_model is None:
        yolov5_tasks._yolov5_model = yolov5_tasks.load_model(
            weights=modelinfo['weights_path'], 
            map_location=map_location
        )

    any_detections = check_if_any_detection_present(
        yolov5_tasks._yolov5_model, file.read(), modelinfo, map_location
    )

    return {"has_detection": any_detections}, 200