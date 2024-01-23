# AI TOOLS USE
## Getting Started

1. Clone this repository.
2. Start a virtualenv with python3.11: `python -m venv .venv`
3. Access env: `source env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy config_example.yml to config.yml
6. Update config.yml with values appropriate to your machine
7. Point environment variable `KLEIN_CONFIG` to the config file: `export KLEIN_CONFIG=/a/path/config.yml`.
8. create an API_KEY from [OPENAI](https://platform.openai.com/api-keys)
9. create .env in the erbium-app repo, example of an API-KEY in the .env file `OPENAI_API_KEY="s607SZ0ufIvj2SSE345646445Sfhshshshshww"`
10. Make sure you have sqlite installed and [Ollama](https://ollama.ai) downloaded on your machine and running, before running any of the commands
11. After downloading Ollama , pull the model parameter used in the erbium-ai.py or llm_sql_agent in this case llama2
    run this command `ollama run llama2`
11. To run each application cd to src folder then run each file you want results from.
    `python3 erbium-ai.py` or `python3 llm_sql_agent`