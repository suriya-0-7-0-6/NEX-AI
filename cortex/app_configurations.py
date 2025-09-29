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

    UPLOADS_DIR = os.path.join(PROJECT_DIR, 'mounts', 'uploads')
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    DATASETS_DIR = os.path.join(PROJECT_DIR, 'mounts', 'datasets')
    os.makedirs(DATASETS_DIR, exist_ok=True)

    LOGS_DIR = os.path.join(PROJECT_DIR, 'mounts', 'logs')
    os.makedirs(LOGS_DIR, exist_ok=True)

    if os.getenv("LOCAL_DEV", "false").lower() == "true":
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    else:
        REDIS_HOST = os.getenv("REDIS_HOST", "redis")


    # REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")

    SOCKETIO_MESSAGE_QUEUE = f'redis://{REDIS_HOST}:{REDIS_PORT}/0'

    CELERY = {
        'broker_url': f'redis://{REDIS_HOST}:{REDIS_PORT}/0',
        'result_backend': f'redis://{REDIS_HOST}:{REDIS_PORT}/0',
        'task_serializer': 'json',
        'result_serializer': 'json',
        'accept_content': ['json'],
        'task_track_started': True,
    }

class ProductionConfigurations(BaseConfigurations):
    SESSION_COOKIE_SECURE = True