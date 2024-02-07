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
9. Add the API_KEY to the config.yml file. You might need to export the key as well using `export OPENAI_API_KEY=sgfkjgkjfhkdjfhb`.
10. For the llama-index & openn-api examples add the config for the data import folder, persistence folder & yaml folder.
11. There is a sample data & yaml file in the data folder, you can use that to test the llama-index & open-api examples. Note that the open-api example doesn't really work that well.
12. Make sure you have sqlite installed and [Ollama](https://ollama.ai) downloaded on your machine and running, before running any of the commands
13. After downloading Ollama , pull the model parameter used in the erbium-ai.py or llm_sql_agent in this case llama2
    run this command `ollama run llama2`
14. To run each application, run one of the following commands in the root directory of the repository
    * `python -m src.erbium-app.erbium-ai`
    * `python -m src.erbium-app.llm_sql_agent`
    * `python -m src.erbium-app.llama-index` - uses openAI
    * `python -m src.erbium-app.llama-index-open` - uses llama2
    * `python -m src.erbium-app.langchain-agent conversation` - uses openAI
    * `python -m src.erbium-app.langchain-agent retrieval` - uses openAI
    * `python -m src.erbium-app.langchain-agent agent` - uses openAI
    * `python -m src.erbium-app.llama-pg-vector --method index` - or `--method query --query "Ask a question about the indexed docs"`. Needs a running pg db with pgvector installed.