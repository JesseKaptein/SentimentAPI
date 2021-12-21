"""
Creates and runs a Flask application with a main homepage and API for sentiment predictions.
"""

import os

from flask import Flask


def create_app():

    # Initiate Flask app
    app = Flask(__name__)

    # Set default app configs
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['CSRF_ENABLED'] = True
    app.config['STATIC_FOLDER'] = os.path.dirname(os.path.realpath(__file__)) + '\\static\\'

    with app.app_context():
        from views import app as main_blueprint
        app.register_blueprint(main_blueprint)
    return app


application = create_app()


if __name__ == "__main__":
    application.run()
