import os
import uuid
from werkzeug.utils import secure_filename
from flask import current_app

def save_uploaded_file(file_storage):
    """Saves an uploaded file to the temporary directory with a unique name."""
    if not file_storage or file_storage.filename == '':
        raise ValueError("No file selected or file has no name.")

    # Sanitize filename and create a unique name to avoid conflicts/overwrites
    original_filename = secure_filename(file_storage.filename)
    _, extension = os.path.splitext(original_filename)
    unique_filename = f"{uuid.uuid4()}{extension}"
    save_path = os.path.join(current_app.config['TEMP_FOLDER'], unique_filename)

    try:
        file_storage.save(save_path)
        current_app.logger.info(f"File saved temporarily to: {save_path}")
        return save_path, original_filename
    except Exception as e:
        current_app.logger.error(f"Error saving file {original_filename}: {e}")
        raise IOError(f"Could not save file: {e}")

def cleanup_file(file_path):
    """Removes a file if it exists."""
    if file_path and os.path.isfile(file_path):
        try:
            os.remove(file_path)
            current_app.logger.info(f"Cleaned up temporary file: {file_path}")
        except OSError as e:
            # Log error but don't raise, cleanup failure is not critical usually
            current_app.logger.error(f"Error removing file {file_path}: {e}")

def get_file_extension(filename):
    """Safely gets the lowercase file extension."""
    _, extension = os.path.splitext(filename)
    return extension.lower()