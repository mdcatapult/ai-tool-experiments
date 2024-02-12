from langchain_community.agent_toolkits.openapi import planner
from langchain.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_openai import ChatOpenAI
import os
import yaml

from config.config import OPENAI_API_KEY, OPENAPI_YAML_DIR

class OpenAPITest:

    def start(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        openai_requests_wrapper = RequestsWrapper(headers=headers)
        with open(os.path.join(OPENAPI_YAML_DIR, "tomics_api.yaml")) as f:
            raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)
            openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
        llm = ChatOpenAI(model_name="gpt-4", temperature=1.0, max_tokens=1000)
        openai_agent = planner.create_openapi_agent(
            openai_api_spec, openai_requests_wrapper, llm, agent_executor_kwargs={"handle_parsing_errors": True}
        )
        # user_query = """I want to perform differential expression analysis on my data for Lungs, Liver & Kidney. What tools can I use?
        # I do not want to delete any data.
        # The data object path for the API call is '/Users/ian.dunlop/atlas-data-clone/second-experiment/transcriptomics/geomxdata-post-qc.rds'."""
        # user_query = """I need a heatmap plot of the genes in this dataset '/Users/ian.dunlop/atlas-data-clone/second-experiment/transcriptomics/geomxdata-post-qc.rds'.
        # How would I achieve this?"""
        user_query = """I need a umap pca plot using this dataset '/Users/ian.dunlop/atlas-data-clone/second-experiment/transcriptomics/geomxdata-post-qc.rds'.
        How would I achieve this?"""
        openai_agent.invoke(user_query)

if __name__ == "__main__":
    openapi_test = OpenAPITest()
    result = openapi_test.start()
    print(result)