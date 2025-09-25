from flask import Blueprint, request, current_app, render_template, jsonify
import importlib
import os
from flask import Response

from cortex.master_orchestrator import fetch_all_problem_ids, fetch_configs, fetch_all_model_archs
from cortex.forms import InferenceForm, TrainForm, PrepareDatasetForm
from cortex.extensions import logger


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



@ml_api.route('/ai_prepare_dataset', methods=['GET', 'POST'])
def prepare_dataset():
    prepare_dataset_form = PrepareDatasetForm()
    prepare_dataset_form.models_list.choices = fetch_all_model_archs()

    train_form = TrainForm()
    train_form.models_list.choices = fetch_all_model_archs()

    inference_form = InferenceForm()
    inference_form.problem_id.choices = fetch_all_problem_ids()

    if request.method == 'POST':
        if prepare_dataset_form.validate_on_submit():
            info = "Form submitted successfully"
            dnnarch = prepare_dataset_form.models_list.data
            input_images_folder_path = prepare_dataset_form.input_images_folder_path.data
            input_annotations_folder_path = prepare_dataset_form.input_annotations_folder_path.data

            if not input_images_folder_path or not input_annotations_folder_path:
                return {'error': 'Input folder paths are required'}, 400
        
            if not os.path.isdir(input_images_folder_path):
                return {'error': f'Input images folder path does not exist: {input_images_folder_path}'}, 400
            
            if not os.path.isfile(input_annotations_folder_path):
                return {'error': f'Input annotations file does not exist: {input_annotations_folder_path}'}, 400

            if not dnnarch:
                return {'error': 'Please select a valid model'}, 400
            
            try:
                tasks_module = importlib.import_module(f"cortex.master_orchestrator.{dnnarch}.tasks")
                prepare_dataset_task = getattr(tasks_module, 'prepare_dataset')
            except (ImportError, AttributeError) as e:
                return {'error': f"Failed to load prepare_dataset task: {e}"}, 500
            
            output_folder = os.path.join(current_app.config['DATASETS_DIR'] + "/" + dnnarch)
            os.makedirs(output_folder, exist_ok=True)

            output = prepare_dataset_task.apply_async(args=[input_images_folder_path, input_annotations_folder_path, output_folder, 0])

            return render_template(
                'ai_pages/ai_inference.html',
                inference_form=inference_form,
                train_form=train_form,
                prepare_dataset_form=prepare_dataset_form,
            ), 200
        
    return render_template(
        'ai_pages/ai_inference.html',
        inference_form=inference_form,
        train_form=train_form,
        prepare_dataset_form=prepare_dataset_form,
    ), 200



@ml_api.route('/ai_inference', methods=['GET', 'POST'])
def ai_inference():
    prepare_dataset_form = PrepareDatasetForm()
    prepare_dataset_form.models_list.choices = fetch_all_model_archs()

    train_form = TrainForm()
    train_form.models_list.choices = fetch_all_model_archs()

    inference_form = InferenceForm()
    inference_form.problem_id.choices = fetch_all_problem_ids()

    if request.method == 'POST':
        if inference_form.validate_on_submit():
            info = "Form submitted successfully"
            problem_id = inference_form.problem_id.data
            upload_image_bytes_file = inference_form.upload_image.data.read()

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

            return render_template(
                'ai_pages/ai_inference.html',
                inference_form=inference_form,
                train_form=train_form,
                prepare_dataset_form=prepare_dataset_form,
            ), 200

    return render_template(
        'ai_pages/ai_inference.html',
        inference_form=inference_form,
        train_form=train_form,
        prepare_dataset_form=prepare_dataset_form,
    ), 200
    

@ml_api.route('/ai_train', methods=['GET', 'POST'])
def ai_train():
    prepare_dataset_form = PrepareDatasetForm()
    prepare_dataset_form.models_list.choices = fetch_all_model_archs()

    train_form = TrainForm()
    train_form.models_list.choices = fetch_all_model_archs()

    inference_form = InferenceForm()
    inference_form.problem_id.choices = fetch_all_problem_ids()

    logger.info(f"Train form validation status: {train_form.validate_on_submit()}")
    logger.info(f"Train form errors: {train_form.errors}")
    
    if request.method == 'POST':
        if train_form.validate_on_submit():
            dnnarch = train_form.models_list.data
            epochs = train_form.epochs.data
            imgsz = train_form.imgsz.data
            batch_size = train_form.batch_size.data
            dataset_yaml_path = train_form.dataset_yaml_path.data
            experiment_name = train_form.experiment_name.data

            if not dnnarch:
                return {'error': 'Please select a valid model'}, 400
            
            if not dataset_yaml_path:
                return {'error': 'Please upload a valid dataset yaml file'}, 400
            
            if not experiment_name:
                return {'error': 'Please enter a valid experiment name'}, 400

            try:
                tasks_module = importlib.import_module(f"cortex.master_orchestrator.{dnnarch}.tasks")
                train_task = getattr(tasks_module, 'train')
            except (ImportError, AttributeError) as e:
                return {'error': f"Failed to load train task: {e}"}, 500

            output = train_task.apply_async(args=[
                dataset_yaml_path, epochs, imgsz, batch_size, "cuda", experiment_name
            ])

            return render_template(
                'ai_pages/ai_inference.html',
                inference_form=inference_form,
                train_form=train_form,
                prepare_dataset_form=prepare_dataset_form,
            ), 200
        
    return render_template(
        'ai_pages/ai_inference.html',
        inference_form=inference_form,
        train_form=train_form,
        prepare_dataset_form=prepare_dataset_form,
    ), 200



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


@ml_api.route('/get_detections', methods=['POST'])
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
