# imports from langchain community and langchain core packages
from langchain.agents import (
    Tool,
    AgentOutputParser,
)
from langchain.agents.agent_types import AgentType
from langchain.agents.tools import Tool
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain.schema import AgentAction, AgentFinish
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.pydantic_v1 import BaseModel, Field
from langchain.sql_database import SQLDatabase


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
import re
from sqlalchemy import create_engine, Result
from typing import List, Dict, Optional, Type, Union, Sequence, Any


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


def ToolExceptionError(s: str) -> None:
    raise ToolException("The tool is not available.")


def _handle_error(error) -> str:
    return str(error)[:50]


# create a DuckDuckGoSearchRun class that inherits from StructuredTool
def query_examples() -> List[Dict]:
    """Example queries and a text that describes what they represent. Use these to train the db LLM tool"""
    return [
        {
            "input": "List all the chemicals in the chemical table.",
            "query": "SELECT * FROM coshh.chemical;",
        },
        {
            "input": "What is the total quantity of chemical in the chemical table?",
            "query": "SELECT CAST(SUBSTRING(quantity, 1, POSITION('L' IN quantity)-1) AS NUMERIC) FROM chemical WHERE chemical_name LIKE '%chemical% ';",
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


# get the input from the user and check if it contains only spaces and non-alphanumeric charactersz
def get_non_alphanumeric_input() -> str:
    while True:
        user_input = input("Enter your text (non-numeric characters allowed): ")

        # Check if the input contains only spaces and non-alphanumeric characters
        if all(word.isalpha() or word.isspace() for word in user_input):
            return user_input
        else:
            print("Invalid input. Please enter only spaces and non-numeric characters.")
            check_response = input("Do you want to continue? (y/n): ")
            if check_response.lower() != "y":
                exit()
            else:
                continue


# create a Table class that inherits from BaseModel
class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(..., description="Name of table in SQL database.")


#  create a QuantityQueryInput class that inherits from BaseModel
class QuantityQueryInput(BaseModel):
    quantity_column: str = Field(
        ...,
        description="A query which has the chemical_name and a Name of a column called quantity in the chemical table. if query return values \
        in g, mg, kg, l, ml, gal, the tool should convert the values to ml. {allowed_values: quantity, chemical_name}",
    )


# the search tool to search hazardous chemicals
class SearchTool(BaseModel):
    name = "search-tool"
    description = "useful for when you need to answer questions about which chemicals in the database is hazardous. \
                   the user will ask targeted questions like 'what chemicals are the least or most hazardous ?' \
                   You should ask targeted questions."

    """Use the tool. when you need to answer questions about which chemicals in the database is hazardous. """

    def run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        search = DuckDuckGoSearchRun()
        return search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("search-tool does not support async")


# create a tool to calculate the quantity of chemicals in the database
class CalculateQuantityColumnTool(BaseModel):
    """Calculate the quantity of chemicals in the database."""

    name = "CalculateQuantityColumnTool to calculate the {quantity} of chemicals in the database"
    description = " I can query the quantity column in the chemical table to get the total quantity of a specific chemical or chemicals. \
                    If the user wants to know the total quantity of a specific chemical is in the database, \
                    this tool can be used to calculate the quantity of chemicals in the database.\
                    allowable values: {input} \
                    useful for when you need to answer questions about the quantity of chemicals in the database and \
                    the quantity of a specific chemical or set of chemicals"
    args_schema: Type[BaseModel] = QuantityQueryInput

    def convert_to_mL(self, volume_strings) -> int:
        total_milliliters = 0
        try:
            digits = re.findall(r"(\d\.\d+[A-Za-z]+)", volume_strings)
        except TypeError:
            digits = volume_strings

        for volume_string in digits:
            print(volume_string)
            quantity_column_match = re.findall(r"(\d+\.\d+[A-Za-z]+)", volume_string)
            for volume_match in quantity_column_match:
                numerical_value_search = re.match(r"([\d.]+)([A-Za-z]+)", volume_match)
                if numerical_value_search:
                    numerical_value = float(numerical_value_search.group(1))
                    metric_unit = numerical_value_search.group(2).lower()
                    if metric_unit == "l":
                        total_milliliters += (
                            numerical_value * 1000
                        )  # Convert liters to milliliters
                    elif metric_unit == "g":
                        total_milliliters += numerical_value * 1000
                    elif metric_unit == "ml":
                        total_milliliters += numerical_value  # Already in milliliters
                    elif metric_unit == "gal":
                        total_milliliters += (
                            numerical_value * 3785.41
                        )  # Convert gallons to milliliters
                    else:
                        raise ValueError(
                            "Unsupported unit. Only 'L', 'mL', and 'gal' are supported."
                        )
                else:
                    raise ValueError("Invalid volume string format.")
        return total_milliliters

    # the call method to calculate the quantity of chemicals in the database
    def __call__(self, data: str) -> int:
        query = f"SELECT quantity FROM chemical WHERE chemical_name IN ('{data}')"
        result = db.run_no_throw(query)
        print("\nresult", result)
        res = self.convert_to_mL.__call__(result)
        print("res", res)
        return res
        # return sum(count)


class SystemMessageAndPromptTemplate:
    """Create the prompt and template for the SQL agent to use"""

    def __init__(self):
        self.system_prefix = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting examples in the database.
       
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to {tools} for interacting with the database. Only use the given tools. Only use the information returned by the tools to construct your final answer.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        Use the following format for your query:
      
        Thought: You should think about what to do, and if the query gives this kind of error  Error: (psycopg2.errors.UndefinedFunction) function sum(character varying) does not exist or  \ 
        invalid input syntax for type numeric: "100ml"
        Try using a tool to calculate the quantity of a specific chemical or set of chemicals, pass the chemical name and the quantity column to the tool

        Action: You must only use the tools , if you encounter an error mentioned in Thought or the question requires you 
        to search the internet such as which is the most hazardous chemical. The action to take, should be one of [{tool_names}]

        Action Input:  the input to the action
        Obeservation:  the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Final Answer: the final answer to the original input question

        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        Question: {input}
        {agent_scratchpad}

        If the question does not seem related to the database, just return "I don't know" as the answer.


        Here are some examples of user inputs and their corresponding SQL queries:"""

    # create a prompt for the SQL agent
    def prompt(self) -> FewShotPromptTemplate:
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=query_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input} \n SQL query: {query}"
            ),
            input_variables=["input", "dialect", "top_k", "tool_names", "tools"],
            prefix=self.system_prefix,
            suffix="",
        )
        return few_shot_prompt

    # create a full prompt for the SQL agent
    def full_prompt(self) -> ChatPromptTemplate:
        full_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=self.prompt()),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return full_prompt


# create a custom output parser for the SQL agent
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Parse the output from the language model
        # Return an AgentAction or AgentFinish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # parse the output from the action and action input
        # regex = r"Action: (.*)[\n]*Action Input: [\s]*([\s\S]*?)[\n]*Obeservation: [\s]*([\s\S]*?)[\n]*"
        regex = r"Action: (.*)[\n]*Action Input: [\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        print("match", match)

        # if it cant be parse the output should raise an error
        if not match:
            raise ValueError(f"Could not parse the output: {llm_output}")

        # get the action and action input
        action, action_input = match.group(1).strip(), match.group(2).strip()

        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


# get the tools that will be used by the SQL agent
def get_function_tools() -> List[Tool]:
    tools = [
        Tool(
            name="SearchTool to search hazardous chemicals",
            func=SearchTool.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="CalculateQuantityColumnTool to calculate the quantity of chemicals in the database",
            func=CalculateQuantityColumnTool().__call__,
            description="useful for when you need to answer questions about the quantity of chemicals in the database and \
                         the quantity of a specific chemical or set of chemicals",
            handle_tool_error=True,
        ),
    ]
    return tools


# using tools, the LLM Chain and the SQL agent to interact with the database
tool_names = [tool.name for tool in get_function_tools()]


# main function
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
                "tool_names": tool_names,
                "tools": get_function_tools(),
            }
        )
    )

    try:

        # create an sql agent executor
        agent = create_sql_agent(
            toolkit=toolkit,
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            extra_tools=get_function_tools(),
            prompt=prompt_val,
            early_stopping_method="\n Observation: ",
            input_variables=[
                "input",
                "dialect",
                "intermediate_steps",
                "top_k",
                "tool_names",
                "tools",
            ],
            max_execution_time=120,
            max_iterations=30,
            # agent_executor_kwargs={"return_intermediate_steps": True},
            handle_parsing_errors=_handle_error,
        )

        query = agent.invoke(
            {
                "input": get_non_alphanumeric_input(),
                "dialect": "SQL",
                "top_k": 5,
                "tool_names": "search-tool, calculate-quantity-column-tool",
                "tools": "search-tool, calculate-quantity-column-tool",
            }
        )

        print("query", query)
        print("\nWould like to run another query? (y/n): ", end="")
        if input().lower() == "y":
            main()
        else:
            print("Goodbye from your friendly SQL agent! :) ")
            exit()

    except Exception as e:
        print("An error occurred: ", e)
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
