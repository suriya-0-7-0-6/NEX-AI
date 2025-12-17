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



@ml_api.route('/predict', methods=['POST'])
def get_detections():
    import torch
    import numpy as np
    import cv2
    import os
    from flask import jsonify
    from cortex.master_orchestrator.yolov8.utils import load_model
    from werkzeug.utils import secure_filename

    file = request.files.get("image")
    problem_id = request.form.get("problem_id")

    if not file or not problem_id:
        return {"error": "Missing image or problem_id"}, 400

    modelinfo = fetch_configs(problem_id)
    if not modelinfo:
        return {"error": "Model configuration not found"}, 404

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(weights=modelinfo["weights_path"], map_location=device)

    file_bytes = file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image file"}, 400

    # Run inference
    results = model.predict(
        source=img,
        imgsz=modelinfo.get("input_size", 640),
        conf=modelinfo.get("confidence_threshold", 0.25),
        save=False,
        device=device
    )

    filename = secure_filename(file.filename)
    output_json = {
        filename: {
            "filename": filename,
            "size": "-1",
            "regions": [],
            "file_attributes": {}
        }
    }

    if results and hasattr(results[0], 'obb') and results[0].obb is not None:
        obb_data = results[0].obb
        xyxyxyxy = obb_data.xyxyxyxy.cpu().numpy()
        confs = obb_data.conf.cpu().numpy()
        clss = obb_data.cls.cpu().numpy()
        class_names = results[0].names

        for i, pts in enumerate(xyxyxyxy):
            # Ensure we work with flat 8-length arrays
            pts = np.array(pts).flatten()
            if pts.shape[0] != 8:
                continue  # skip invalid ones

            all_points_x = [int(round(pts[j])) for j in range(0, 8, 2)]
            all_points_y = [int(round(pts[j])) for j in range(1, 8, 2)]

            class_id = int(clss[i])
            label = class_names[class_id]

            region = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y
                },
                "region_attributes": {
                    "Label": label
                }
            }
            output_json[filename]["regions"].append(region)

    return jsonify(output_json), 200