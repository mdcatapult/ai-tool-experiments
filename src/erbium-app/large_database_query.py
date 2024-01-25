# imports from langchain community and langchain core packages
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

# local imports and python builtins
from open_ai_config.openai_config import OPENAI_API_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


# Replace "sqlite:///MDC.DB" with your actual database URI
db_uri = "sqlite:///erbium-app/MDC.db"

# instantiate the SQLDatabase object
db = SQLDatabase.from_uri(db_uri)


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(..., description="Name of table in SQL database.")
    columns: list = Field(..., description="Columns in table in SQL database.")
    rows: list = Field(..., description="Rows in table in SQL database.")


table_names = "\n".join(db.get_usable_table_names())
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

# Agent roles are defined and instantiated here
if __name__ == "__main__":
    table_chain = create_extraction_chain_pydantic(Table, llm, system_message=system)
    table_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"})
