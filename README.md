# AI TOOLS USE

## About
Lots and lots of [LLM](https://en.wikipedia.org/wiki/Large_language_model) [agent](https://en.wikipedia.org/wiki/Intelligent_agent) and [RAG](https://en.wikipedia.org/wiki/Prompt_engineering#Retrieval-augmented_generation) demos using [crew-ai](https://github.com/joaomdmoura/crewAI), [langchain](https://www.langchain.com) & [llama-index](https://docs.llamaindex.ai/en/stable/index.html). 
They are not supposed to be robust and may not always work.
They are there to demonstrate the capabilities of the tools and to provide a starting point for further development.

## Getting Started

1. Clone this repository.
2. Create a virtualenv with python3.11: `virtualenv -p python3.11 venv` (or your favourite way of doing it)
3. Activate the python environment: `source env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy config_example.yml to config.yml
6. Update config.yml with values appropriate to your machine. See the config_example.yml & steps 9/10 for more details.
7. Point environment variable `KLEIN_CONFIG` to the config file: `export KLEIN_CONFIG=/a/path/config.yml`.
8. create an API_KEY from [OPENAI](https://platform.openai.com/api-keys)
9. Add the API_KEY to the config.yml file. You might need to export the key as well using `export OPENAI_API_KEY=sgfkjgkjfhkdjfhb`.
10. For the llama-index & open-api examples add the config for the data import folder, persistence folder & yaml folder.
11. There is a sample openAPI yaml file in the data folder, you can use that to test the open-api examples. For the llama-index tests you will need to put some docs files in a folder. Note that the open-api example doesn't really work that well. It is based on the [Transcriptomics API](https://gitlab.com/medicines-discovery-catapult/informatics/dsp-atlas/dsp-atlas-transcriptomics-api) with some changes made to persuade the code to run.
12. Make sure you have [sqlite installed](https://www.sqlite.org/download.html) (`brew install sqlite` should do it on a mac) and [Ollama](https://ollama.ai) downloaded on your machine and running, before running any of the commands
13. After downloading Ollama download the llama2 model used in the ai-insights-agent and the medicines-discovery-agents llm_sql_agent by running this command `ollama run llama2`. You can use this llm model in any of the examples but you would need to change the code.
14. To run each application, run one of the following commands in the root directory of the repository
    * `python -m src.app.crewai_examples.ai_insights_agent` - uses openAI. Uses crew-ai agents to converse with each other and gather facts around AI based topics
    * `python -m src.app.crewai_examples.medicines_discovery_agents` - uses openAI. Uses crew-ai agents to converse with each other and gather facts around drug discovery based topics
    * `python -m src.app.llama_examples.llama_index` - uses openAI. Vectorises documents into a local file based store and runs a query over the stored documents. If it complains about embedding token size then delete the existing data persistence dir so it can index from the start 
    * `python -m src.app.llama_examples.llama_index_open` - uses llama2. Uses open source model to vectorise documents and runs a query over the stored documents. If it complains about embedding token size then delete the existing data persistence dir so it can index from the start
    * `python -m src.app.llama_examples.llama_pg_vector --method index` - or `--method query --query "Ask a question about the indexed docs"`. Vectorise docs using pgvector and query over them. Needs a running pg db with pgvector installed. See section below.
    * `python -m src.app.langchain_examples.llm_sql_agent` - uses openAI. Trains an llm about a specific sql schema that you can then free text query over.
    * `python -m src.app.langchain_examples.langchain_agent conversation` - uses openAI. Demonstrates a chain of prompts enhanced with a web page as RAG
    * `python -m src.app.langchain_examples.langchain_agent retrieval` - uses openAI. Demonstrates single prompt enhanced with a web page as RAG
    * `python -m src.app.langchain_examples.langchain_agent agent` - uses openAI. Demonstrates using a tool (DuckDuckGo) along with a prompt to answer some simple questions.
    * `python -m src.app.langchain_examples.open_api_langchain` - uses openAI. Use the DSP atlas openAPI to query the database. This is a bit of a mess and doesn't really work.
    * `python -m src.app.langchain_examples.using_tools_langchain_agent` - uses openAI. Demonstrates how to use a free text query along wih an sql agent to fetch information from an sql database. The code trains the LLM with example sql queries against the coshh db. Ask SE for login details.

## Starting the PostgreSQL vector database

1. execute docker compose:
```bash
cd postgres_files
docker-compose up --build
```
P.S. If used before then you may need to remove existing volumes Either `docker-compose down -v` or `docker volume ls` and `docker volume rm the-volume-name` before running the command above.

2. create a Table on PSQL
```bash
docker exec -it postgres_files-postgres-1 /bin/bash
psql -U postgres
```

Then:

```sql
CREATE TABLE cro_vector_db (
    id bigserial PRIMARY KEY,
    id_cro VARCHAR(50),
    cap_description TEXT,
    capabilities_vector vector(768)-- number of dimensions
);
```
3. Manually, create the extension on PSQL if it doesn't already exist: `CREATE EXTENSION IF NOT EXISTS vector;` Then exit the psql shell and the docker image.
