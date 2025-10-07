from flask import Blueprint, request, current_app, render_template, jsonify
import importlib
import os
from flask import Response
import time
from cortex.master_orchestrator import fetch_all_problem_ids, fetch_configs, fetch_all_model_archs
from cortex.forms import InferenceForm, TrainForm, PrepareDatasetForm, BulkInferenceForm
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

    bulk_inference_form = BulkInferenceForm()
    bulk_inference_form.problem_id.choices = fetch_all_problem_ids()    

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
                output_folder_path = os.path.join(current_app.config['DATASETS_DIR'], dnnarch, f"prepare_dataset_{int(time.time())}")
            except (ImportError, AttributeError) as e:
                return {'error': f"Failed to load prepare_dataset task: {e}"}, 500

            output = prepare_dataset_task.apply_async(args=[input_images_folder_path, input_annotations_folder_path, output_folder_path, 0])

            return render_template(
                'ai_pages/ai_inference.html',
                inference_form=inference_form,
                bulk_inference_form=bulk_inference_form,
                train_form=train_form,
                prepare_dataset_form=prepare_dataset_form,
            ), 200
        
    return render_template(
        'ai_pages/ai_inference.html',
        inference_form=inference_form,
        bulk_inference_form=bulk_inference_form,
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

    bulk_inference_form = BulkInferenceForm()
    bulk_inference_form.problem_id.choices = fetch_all_problem_ids() 

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
                output_folder_path = os.path.join(current_app.config['LOGS_DIR'], "single_image_inferences")
            except (ImportError, AttributeError) as e:
                return {'error': f"Failed to load detect task: {e}"}, 500

            detect_task.apply_async(args=[upload_image_bytes_file, modelinfo, output_folder_path])

            return render_template(
                'ai_pages/ai_inference.html',
                inference_form=inference_form,
                bulk_inference_form=bulk_inference_form,
                train_form=train_form,
                prepare_dataset_form=prepare_dataset_form,
            ), 200

    return render_template(
        'ai_pages/ai_inference.html',
        inference_form=inference_form,
        bulk_inference_form=bulk_inference_form,
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

    bulk_inference_form = BulkInferenceForm()
    bulk_inference_form.problem_id.choices = fetch_all_problem_ids()

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
                output_folder_path = os.path.join(current_app.config['LOGS_DIR'], "train_results")
            except (ImportError, AttributeError) as e:
                return {'error': f"Failed to load train task: {e}"}, 500

            output = train_task.apply_async(args=[
                dataset_yaml_path, epochs, imgsz, batch_size, experiment_name, output_folder_path
            ])

            return render_template(
                'ai_pages/ai_inference.html',
                inference_form=inference_form,
                bulk_inference_form=bulk_inference_form,
                train_form=train_form,
                prepare_dataset_form=prepare_dataset_form,
            ), 200
        
    return render_template(
        'ai_pages/ai_inference.html',
        inference_form=inference_form,
        bulk_inference_form=bulk_inference_form,
        train_form=train_form,
        prepare_dataset_form=prepare_dataset_form,
    ), 200



@ml_api.route('/predict', methods=['GET', 'POST'])
def predict():
    problem_id = request.form.get("problem_id")
    image = request.files.get("image")
     
    if not problem_id:
        return {'error': 'Problem ID is required'}, 400

    if not image or not problem_id:
        return {"error": "Missing folder_path or problem_id"}, 400
        
    modelinfo = fetch_configs(problem_id)

    dnnarch = modelinfo['dnnarch']
    tasks_module = importlib.import_module(f"cortex.master_orchestrator.{dnnarch}.tasks")
    bulk_inferencet_task = getattr(tasks_module, 'bulk_inference')
    detection_results = bulk_inferencet_task(image, modelinfo)
    return detection_results
