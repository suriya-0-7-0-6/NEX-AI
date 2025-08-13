from flask import Flask
from cortex.app_configurations import BaseConfigurations, ProductionConfigurations

from cortex.extensions import init_extensions
from cortex.ml_routes import ml_api
from cortex.files_manager import files_api

def create_app():
    app = Flask(
        __name__,
        template_folder = BaseConfigurations.TEMPLATES_DIR,
        static_folder = BaseConfigurations.STATIC_DIR
    )

    app.config.from_object(BaseConfigurations)

    app.register_blueprint(ml_api)
    app.register_blueprint(files_api)

    init_extensions(app)

    return app