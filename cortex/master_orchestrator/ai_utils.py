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