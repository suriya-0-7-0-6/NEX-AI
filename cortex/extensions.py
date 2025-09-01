from celery import Celery
from flask_socketio import SocketIO

celery = Celery(__name__)
socketio = SocketIO(cors_allowed_origins='*')

def init_extensions(app):
    socketio.init_app(app, message_queue=app.config['SOCKETIO_MESSAGE_QUEUE'])
    
    @socketio.on('connect')
    def handle_connect():
        print("Client connected")
    
    init_celery(celery, app)


def init_celery(celery, app):
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery.Task = ContextTask
    celery.config_from_object(app.config['CELERY'])
    celery.autodiscover_tasks(['cortex.master_orchestrator.yolov7.tasks', 'cortex.master_orchestrator.yolov8.tasks'])