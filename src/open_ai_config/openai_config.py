from klein_config import get_config

config = get_config()

# Define the OpenAI API key
OPENAI_API_KEY = config.get("open_ai.open_api_key")
DATA_IMPORT_DIRECTORY = config.get("data_directory")
DATA_PERSIST_DIRECTORY = config.get("data_persist_directory")
