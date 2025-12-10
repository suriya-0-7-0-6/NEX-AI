from flask import Blueprint, request, current_app, render_template, jsonify
import importlib
import os
from flask import Response
import time
from cortex.master_orchestrator import fetch_all_problem_ids, fetch_configs, fetch_all_model_archs, fetch_model_specific_training_configs, fetch_problem_id_specific_inference_configs, fetch_model_specific_prepare_dataset_configs
from cortex.forms import DynamicAIForm
from cortex.extensions import logger


ml_api = Blueprint('ml_routes', __name__)

@ml_api.route('/get_all_problem_ids', methods=['GET'])
def get_all_problem_ids():
    all_problem_ids = fetch_all_problem_ids()
    return {'problem_ids': all_problem_ids}, 200

@ml_api.route('/get_all_model_archs', methods=['GET'])
def get_all_model_archs():
    all_model_archs = fetch_all_model_archs()
    return {'all_model_archs': all_model_archs}, 200

@ml_api.route('/get_train_form_fields/<model_arch>', methods=['GET'])
def get_train_form_fields(model_arch):
    training_configs = fetch_model_specific_training_configs(model_arch)
    return jsonify(training_configs), 200

@ml_api.route('/get_inference_form_fields/<problem_id>', methods=['GET'])
def get_inference_form_fields(problem_id):
    inference_configs = fetch_problem_id_specific_inference_configs(problem_id)
    return jsonify(inference_configs), 200

@ml_api.route('/get_prepare_dataset_form_fields/<model_arch>', methods=['GET'])
def get_prepare_dataset_form_fields(model_arch):
    prepare_dataset_configs = fetch_model_specific_prepare_dataset_configs(model_arch)
    return jsonify(prepare_dataset_configs), 200

@ml_api.route('/get_problem_config/<problem_id>', methods=['GET'])
def get_problem_config(problem_id):
    configs = fetch_configs(problem_id)
    if not configs:
        return {'error': 'Configuration file is empty'}, 404
    return {'configs': configs}, 200

@ml_api.route('/ai_inference', methods=['GET', 'POST'])
def ai_inference():
    dynamic_ai_form = DynamicAIForm() 
    if request.method == 'POST':
        if not dynamic_ai_form.validate():
            return {"error": "Invalid CSRF token"}, 400
        params = request.form.to_dict()
        files = {
            name: file.read()
            for name, file in request.files.items()
            if file.filename
        }
        params["files"] = files
        problem_id = params.get("problem_id")
        if not problem_id:
            return {"error": "Problem ID is required"}, 400
        params["modelinfo"] = fetch_configs(problem_id)
        if not params["modelinfo"]:
            return {"error": "Model configuration not found"}, 404
        try:
            dnnarch = params["modelinfo"]["dnnarch"]
            tasks_module = importlib.import_module(
                f"cortex.master_orchestrator.{dnnarch}.tasks"
            )
            detect_task = getattr(tasks_module, "detect")
        except Exception as e:
            return {"error": f"Failed to load detect task: {str(e)}"}, 500
        params["output_folder_path"] = os.path.join(current_app.config["LOGS_DIR"], "single_image_inferences")
        detect_task.apply_async(args=[params])
        return {"status": "Inference started"}, 202

    return render_template('ai_pages/ai_inference.html', dynamic_ai_form=dynamic_ai_form), 200


@ml_api.route("/ai_train", methods=["GET", "POST"])
def ai_train():
    dynamic_ai_form = DynamicAIForm()
    if request.method == "POST":
        if not dynamic_ai_form.validate():
            return {"error": "Invalid CSRF token"}, 400
        params = request.form.to_dict()
        files = {
            name: file.read()
            for name, file in request.files.items()
            if file.filename
        }
        params["files"] = files
        dnnarch = params.get("models_list")
        if not dnnarch:
            return {"error": "Model architecture is required"}, 400
        try:
            tasks_module = importlib.import_module(
                f"cortex.master_orchestrator.{dnnarch}.tasks"
            )
            train_task = getattr(tasks_module, "train")
        except Exception as e:
            return {"error": f"Failed to load train task: {str(e)}"}, 500
        params["dnnarch"] = dnnarch
        params["output_folder_path"] = os.path.join(
            current_app.config["LOGS_DIR"],
            "train_results"
        )
        train_task.apply_async(args=[params])
        return {"status": "Training started"}, 202
    return render_template("ai_pages/ai_inference.html", dynamic_ai_form=dynamic_ai_form)


@ml_api.route("/ai_prepare_dataset", methods=["GET", "POST"])
def ai_prepare_dataset():
    dynamic_ai_form = DynamicAIForm()
    if request.method == "POST":
        if not dynamic_ai_form.validate():
            return {"error": "Invalid CSRF token"}, 400
        params = request.form.to_dict()
        files = {
            name: file.read()
            for name, file in request.files.items()
            if file.filename
        }
        params["files"] = files
        dnnarch = params.get("models_list")
        if not dnnarch:
            return {"error": "Model architecture is required"}, 400
        try:
            tasks_module = importlib.import_module(
                f"cortex.master_orchestrator.{dnnarch}.tasks"
            )
            prepare_dataset_task = getattr(tasks_module, "prepare_dataset")
        except Exception as e:
            return {"error": f"Failed to load prepare_dataset task: {str(e)}"}, 500
        params["dnnarch"] = dnnarch
        params["output_folder_path"] = os.path.join(
            current_app.config["LOGS_DIR"],
            "prepare_dataset_results",
            f'{params.get("models_list")}',
            f'{time.time()}'
        )
        prepare_dataset_task.apply_async(args=[params])
        return {"status": "Preparing Dataset"}, 202
    return render_template("ai_pages/ai_inference.html", dynamic_ai_form=dynamic_ai_form)