#  Copyright 2024 Medicines Discovery Catapult
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from klein_config import get_config

config = get_config()

# Define the OpenAI API key
OPENAI_API_KEY = config.get("open_ai.open_api_key")
DATA_IMPORT_DIRECTORY = config.get("data_import_directory")
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
