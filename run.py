import os
from app import create_app

# Create the Flask app instance using the factory function
# It wi ll load configuration based on the default 'app.config.Config'
# or an environment variable like FLASK_APP_CONFIG if you set that up in create_app
app = create_app()

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Use host='0.0.0.0' to make it accessible on your network
    # Debug will be enabled/disabled based on FLASK_DEBUG in .env/Config
    is_debug = app.config.get('FLASK_DEBUG', False)
    print(f"Starting Flask app on host 0.0.0.0, port {port}, Debug: {is_debug}")
    app.run(host='0.0.0.0', port=port, debug=is_debug)

# For Production (using Gunicorn):
# You would typically run: gunicorn --bind 0.0.0.0:5000 "app:create_app()"
# Or: gunicorn --bind 0.0.0.0:5000 run:app (if run.py defines app = create_app())
# Gunicorn handles multiple workers, logging, etc.