import os
from dotenv import load_dotenv

# Load environment variables from .env file, searching upwards from the current file
# This makes it work correctly whether run from root or inside app/
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded .env file from: {dotenv_path}") # Add print statement for verification
else:
    print("Warning: .env file not found at expected location.")


class Config:
    """Base configuration variables."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-default-fallback-secret-key-CHANGE-IN-PROD'
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')

    # -- Embedding Service Config --
    EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY")
    EMBEDDING_API_VERSION = os.environ.get("EMBEDDING_API_VERSION")
    EMBEDDING_API_TYPE = os.environ.get("EMBEDDING_API_TYPE")
    EMBEDDING_API_BASE = os.environ.get("EMBEDDING_API_BASE")
    EMBEDDING_DEPLOYMENT = os.environ.get("EMBEDDING_DEPLOYMENT") # Model name for embeddings

    # -- Language Model Config --
    GPT_API_KEY = os.environ.get("GPT_API_KEY")
    GPT_API_VERSION = os.environ.get("GPT_API_VERSION")
    GPT_API_TYPE = os.environ.get("GPT_API_TYPE")
    GPT_API_BASE = os.environ.get("GPT_API_BASE")
    GPT_DEPLOYMENT = os.environ.get("GPT_DEPLOYMENT") # Model name for completion/chat

    # -- File Paths --
    # Get project root by going up one level from the 'app' directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    TEMP_FOLDER = os.path.join(PROJECT_ROOT, 'tmp')
    CHROMADB_FOLDER = os.path.join(PROJECT_ROOT, 'chromadb')

    # Ensure temp and chromadb directories exist
    @staticmethod
    def initialize_dirs():
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
        os.makedirs(Config.CHROMADB_FOLDER, exist_ok=True)

    # -- Validation checks --
    @staticmethod
    def validate():
        Config.initialize_dirs() # Ensure dirs exist before validating other things

        required_vars = [
            "EMBEDDING_API_KEY", "EMBEDDING_API_BASE", "EMBEDDING_API_TYPE",
            "EMBEDDING_API_VERSION", "EMBEDDING_DEPLOYMENT",
            "GPT_API_KEY", "GPT_API_BASE", "GPT_API_TYPE",
            "GPT_API_VERSION", "GPT_DEPLOYMENT"
        ]
        missing_vars = [var for var in required_vars if not getattr(Config, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file or environment setup.")

# Call validate on import to ensure config is loaded and checked early
try:
    Config.validate()
except ValueError as e:
    print(f"Configuration Error: {e}")
    # Depending on severity, you might want to sys.exit(1) here