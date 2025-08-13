import os

class BaseConfigurations:

    SECRET_KEY = "dev-secret"

    DEBUG = False

    APP_DIR = os.path.dirname(os.path.abspath(__file__))

    PROJECT_DIR = os.path.dirname(APP_DIR)

    MODEL_REGISTRY = os.path.join(APP_DIR, 'modelinfo_registry')

    TEMPLATES_DIR = os.path.join(APP_DIR, 'templates')

    STATIC_DIR = os.path.join(APP_DIR, 'static')

    MODEL_ARCHS_DIR = os.path.join(PROJECT_DIR, 'grey_matter', 'model_archs')

    WEIGHTS_DIR = os.path.join(PROJECT_DIR, 'grey_matter', 'weights')

    LOGS_DIR = os.path.join(PROJECT_DIR, 'grey_matter', 'logs')

    UPLOADS_DIR = os.path.join(PROJECT_DIR, 'uploads')

    RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

    SOCKETIO_MESSAGE_QUEUE = 'redis://localhost:6379/0'

    CELERY = {
        'broker_url': 'redis://localhost:6379/0',
        'result_backend': 'redis://localhost:6379/0',
        'task_serializer': 'json',
        'result_serializer': 'json',
        'accept_content': ['json'],
        'task_track_started': True,
    }

class ProductionConfigurations(BaseConfigurations):
    SESSION_COOKIE_SECURE = True