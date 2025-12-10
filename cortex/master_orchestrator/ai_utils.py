from flask import current_app
import importlib
import os

def fetch_all_problem_ids():
    all_problem_ids = []
    for config_file in os.listdir(current_app.config['MODEL_REGISTRY']):
        if not config_file.endswith(".py") or config_file.startswith("__"):
            continue
        try:
            module_name = os.path.splitext(config_file)[0]
            problem_configurations = importlib.import_module(f"cortex.modelinfo_registry.{module_name}")
            all_problem_ids.append(problem_configurations.CONFIG['id'])
        except ImportError as e:
            print(f"Error importing {module_name}: {e}")
    return all_problem_ids

def fetch_configs(problem_id):
    print(os.path.join(current_app.config['MODEL_REGISTRY'], f"{problem_id}.py"))
    config_file = f"{problem_id}.py"
    try:
        module_name = os.path.splitext(config_file)[0]
        problem_configurations = importlib.import_module(f"cortex.modelinfo_registry.{module_name}")
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")

    return problem_configurations.CONFIG if hasattr(problem_configurations, 'CONFIG') else None


def fetch_model_specific_training_configs(model_arch):
    from cortex.forms.dynamic_form_fields import TRAINING_CONFIGS
    return TRAINING_CONFIGS.get(model_arch, [])

def fetch_model_specific_prepare_dataset_configs(model_arch):
    from cortex.forms.dynamic_form_fields import PREPARE_DATASET_CONFIGS
    return PREPARE_DATASET_CONFIGS.get(model_arch, [])

def fetch_problem_id_specific_inference_configs(problem_id):
    from cortex.forms.dynamic_form_fields import INFERENCE_CONFIGS
    return INFERENCE_CONFIGS.get(problem_id, [])


def fetch_all_model_archs():
    from grey_matter.model_archs.List_of_model_archs import MODEL_ARCHS_LIST
    return MODEL_ARCHS_LIST