from klein_config import get_config

config = get_config()

# Define the OpenAI API key
OPENAI_API_KEY = config.get("open_ai.open_api_key")
DATA_IMPORT_DIRECTORY = config.get("data_directory")
DATA_PERSIST_DIRECTORY = config.get("data_persist_directory")
PROJECT_FILE_PATH = config.get("project_file_path")

# Define the PostgreSQL database connection parameters
DATABASE_PASS = config.get("postgres.database_password")
DATABASE_NAME = config.get("postgres.database_name")
DATABASE_USER = config.get("postgres.database_user")
DATABASE_HOST = config.get("postgres.database_host")
DATABASE_PORT = config.get("postgres.database_port")
DATABASE_SCHEMA_NAME = config.get("postgres.database_schema")

OPENAPI_YAML_DIR = config.get("openapi_yaml_dir")
