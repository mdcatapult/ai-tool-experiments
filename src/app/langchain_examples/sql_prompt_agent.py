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

# imports from langchain community and langchain core packages
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool


# local imports and python builtins
from src.config.config import (
    OPENAI_API_KEY,
    DATABASE_NAME,
    DATABASE_PASS,
    DATABASE_USER,
    DATABASE_HOST,
    DATABASE_PORT,
    DATABASE_SCHEMA_NAME,
)

from logging import getLogger
import os
import psycopg2
from sqlalchemy import create_engine
from typing import List, Dict

# create a logger object
Logger = getLogger(__name__)

# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# postgresql database connection parameters
db_uri = f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
engine = create_engine(
    db_uri,
    connect_args={"options": "-csearch_path={}".format(DATABASE_SCHEMA_NAME)},
)


# chatopenai language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# instantiate the SQLDatabase object
db = SQLDatabase(engine=engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


def query_examples() -> List[Dict]:
    """Example queries and a text that describes what they represent. Use these to train the db LLM tool"""
    return [
        {
            "input": "List all the chemicals in the chemical table.",
            "query": "SELECT * FROM coshh.chemical;",
        },
        {
            "input": "What chemicals are about to expire and what lab and cupboard are they in?",
            "query": "SELECT chemical_name, lab_location, cupboard FROM coshh.chemical WHERE expiry < CURRENT_DATE;",
        },
        {
            "input": "How many chemicals are in the chemical table?",
            "query": "SELECT COUNT(*) FROM coshh.chemical;",
        },
        {
            "input": "What is going to expire this week?",
            "query": "SELECT * FROM coshh.chemical WHERE expiry BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days';",
        },
        {
            "input": "Who last updated the chemical table?",
            "query": "SELECT last_updated_by FROM coshh.chemical ORDER BY last_updated DESC LIMIT 10;",
        },
        {
            "input": "Which chemicals were added to do place a limit on the query?",
            "query": "SELECT * FROM coshh.chemical WHERE added < CURRENT_DATE;",
        },
        {
            "input": "What is the chemical number of the chemical with the name 'Acetone'?",
            "query": "SELECT * FROM chemical WHERE chemical_name LIKE '%Acetone%';",
        },
        {
            "input": "Find chemicals stored at a specific temperature.",
            "query": "SELECT * FROM chemical WHERE storage_temp LIKE '%specific temperature% ';",
        },
        {
            "input": "Find chemicals with a specific CAS number.",
            "query": "SELECT * FROM chemical WHERE cas_number LIKE '%specific cas number%';",
        },
        {
            "input": "Find chemicals with a specific safety data sheet.",
            "query": "SELECT * FROM chemical WHERE safety_data_sheet LIKE '%specific safety data sheet%';",
        },
        {
            "input": "Find chemicals with a specific COSHH link.",
            "query": "SELECT * FROM chemical WHERE coshh_link LIKE '%specific coshh link%';",
        },
        {
            "input": "Find chemicals with a specific photo path.",
            "query": "SELECT * FROM chemical WHERE photo_path LIKE '%specific photo path%';",
        },
        {
            "input": "Find chemicals with a specific chemical owner.",
            "query": "SELECT * FROM chemical WHERE chemical_owner LIKE '%specific chemical owner%';",
        },
        {
            "input": "Find chemicals with a specific project specific.",
            "query": "SELECT * FROM chemical WHERE project_specific LIKE '%specific project specific%';",
        },
        {
            "input": "Find chemicals with a specific cupboard.",
            "query": "SELECT * FROM chemical WHERE cupboard LIKE '%specific cupboard%';",
        },
        {
            "input": "what chemicals or find chemicals which were added recently and who last updated the chemical table?",
            "query": "SELECT * FROM chemical WHERE added > CURRENT_DATE - INTERVAL '30 days' AND last_updated_by is not null;",
        },
        {
            "input": "Find chemicals with a specific lab location.",
            "query": "SELECT * FROM chemical WHERE lab_location LIKE '%specific lab location%';",
        },
        {
            "input": "Find chemicals with a specific matter state.",
            "query": "SELECT * FROM chemical WHERE matter_state LIKE '%specific matter state%';",
        },
        {
            "input": "Find chemicals with a specific quantity.",
            "query": "SELECT * FROM chemical WHERE quantity LIKE '%specific quantity%';",
        },
        {
            "input": "Find chemicals with a specific expiry date.",
            "query": "SELECT * FROM chemical WHERE expiry LIKE '%specific expiry date%';",
        },
        {
            "input": "Find chemicals with a specific added date.",
            "query": "SELECT * FROM chemical WHERE added LIKE '%specific added date%';",
        },
        {
            "input": "search for chemicals by lab location and matter state.",
            "query": "SELECT * FROM chemical WHERE lab_location LIKE '%specific lab location%' AND matter_state LIKE '%specific matter state%';",
        },
        {
            "input": "list or search chemicals assigned to a specific project",
            "query": "SELECT * FROM chemical WHERE project_specific LIKE '%specific project%';",
        },
    ]


# query selector for the semantic similarity example selector
query_selector = SemanticSimilarityExampleSelector.from_examples(
    query_examples(),
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)


def get_non_alphanumeric_input():
    """Get user input and use it to ask questions of the db"""

    while True:
        user_input = input("Enter your text (non-numeric characters allowed): ")

        # Check if the input contains only spaces and non-alphanumeric characters

        if ((not word.isalnum()) for word in user_input.split()):

            return user_input
        else:
            print("Invalid input. Please enter only spaces and non-numeric characters.")
            check_response = input("Do you want to continue? (y/n): ")
            if check_response.lower() != "y":
                print("Goodbye from your friendly SQL agent! :) ")
                exit()
            else:
                continue


# create a Table class that inherits from BaseModel
class Table(BaseModel):
    """Represents a table in SQL database."""

    name: str = Field(..., description="Name of table in SQL database.")


# create a tool to list the tables in the database
@tool("list-table-tool", args_schema=Table)
def get_tables(categories: List[Table]) -> List[str]:
    """Get the tables in the schema."""
    tables = []
    for category in categories:
        if category.name == "chemical":
            tables.extend(
                [
                    "chemical",
                    "chemical_to_hazard",
                ]
            )
        elif category.name == "audit":
            tables.extend(["audit_coshh_logs"])
    return tables


class SystemMessageAndPromptTemplate:
    """Create the prompt and template for the SQL agent to use"""

    def __init__(self):
        self.system_prefix = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the zrelevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If the question does not seem related to the database, just return "I don't know" as the answer.

        Here are some examples of user inputs and their corresponding SQL queries:"""

    def prompt(self):
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=query_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input} \n SQL query: {query}"
            ),
            input_variables=["input", "dialect", "top_k"],
            prefix=self.system_prefix,
            suffix="",
        )
        return few_shot_prompt

    def full_prompt(self):
        full_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=self.prompt()),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return full_prompt


def main():

    # formatted prompt
    prompt_val = (
        SystemMessageAndPromptTemplate()
        .full_prompt()
        .invoke(
            {
                "input": "chemical table",
                "top_k": 5,
                "dialect": "SQLite",
                "agent_scratchpad": [],
            }
        )
    )

    # create an sql agent executor
    agent_executor = create_sql_agent(
        toolkit=toolkit,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        prompt=prompt_val,
        handle_parsing_errors=True,
    )
    try:
        query = agent_executor.invoke({"input": get_non_alphanumeric_input()})
        print(query)
        print("\nWould you like to run another query? (y/n): ", end="")
        if input().lower() == "y":
            main()
        else:
            print("Goodbye from your friendly SQL agent! :) ")
            exit()

    except Exception as e:
        Logger.exception(e)
        exit()


if __name__ == "__main__":
    main()

# potential user questions
# How many chemical are in the chemical table?
# what is going to expire this week?
# What chemicals are about to expire and what lab and cupboard are they in?
# who last updated the chemical table?
# which chemical were added to do place a limit on the query

# what chemical
# 0                  id
# 1          cas_number
# 2       chemical_name
# 3     chemical_number
# 4        matter_state
# 5            quantity
# 6               added
# 7              expiry
# 8   safety_data_sheet
# 9          coshh_link
# 10       lab_location
# 11       storage_temp
# 12        is_archived
# 13   project_specific
# 14           cupboard
# 15         photo_path
# 16     chemical_owner
# 17    last_updated_by
