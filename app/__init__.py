import os
import logging
from logging.handlers import RotatingFileHandler # Import handler
from flask import Flask
from flask import Flask, jsonify
from flask import Blueprint

def create_app(config_class_string='app.config.Config'):
    """Creates and configures the Flask application using the App Factory pattern."""
    app = Flask(__name__)

    # --- Load Configuration ---
    try:
        app.config.from_object(config_class_string)
        # You could add Config.validate() here too if desired
        # Config.validate()
    except ImportError:
        # Use basic logger before app logger is fully configured
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Configuration class '{config_class_string}' not found.")
        raise
    except ValueError as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Configuration validation failed: {e}")
        raise # Stop app creation if config is invalid

    # --- Configure Logging ---
    log_level = logging.DEBUG if app.config.get('FLASK_DEBUG') else logging.INFO

    # Use Flask's built-in logger
    app.logger.setLevel(log_level)

    # --- >> ADD FILE LOGGING CONFIGURATION << ---
    if not app.debug and not app.testing: # Only log to file in production-like environments
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(app.config['PROJECT_ROOT'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'app.log')

        # Use RotatingFileHandler to limit log file size
        # Max 10MB per file, keep 5 backup files
        file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*10, backupCount=5)

        # Set a specific format for file logs
        file_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        file_handler.setFormatter(file_formatter)

        # Set the logging level for the file handler
        file_handler.setLevel(logging.INFO) # Log INFO and above to file

        # Add the file handler to the Flask app's logger
        app.logger.addHandler(file_handler)

        app.logger.info('----- Application Starting -----') # Log startup event to file

    # Console logging (Flask usually adds one by default when debug=True)
    # You could explicitly add a StreamHandler here if needed for more control
    # stream_handler = logging.StreamHandler()
    # stream_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # stream_handler.setFormatter(stream_formatter)
    # stream_handler.setLevel(log_level)
    # app.logger.addHandler(stream_handler) # Add if you want explicit console logs

    # --- End Logging Configuration ---

    app.logger.info("Flask app creating...") # This will now go to file if not in debug mode
    app.logger.debug(f"Debug mode: {app.config.get('FLASK_DEBUG')}") # Debug messages might only go to console
    app.logger.info(f"Temporary Folder: {app.config.get('TEMP_FOLDER')}")
    app.logger.info(f"ChromaDB Folder: {app.config.get('CHROMADB_FOLDER')}")

    # --- Register Blueprints ---
    try:
        from app.routes import qa_bp
        app.register_blueprint(qa_bp, url_prefix='/api/v1')
        app.logger.info("QA Blueprint registered under /api/v1")
    except Exception as e:
        app.logger.error(f"Failed to register blueprints: {e}", exc_info=True)
        raise

    # --- Simple Health Check Route ---
    @app.route('/health')
    def health_check():
        return jsonify({"status": "ok"}), 200

    app.logger.info("Flask app created successfully.")
    return app