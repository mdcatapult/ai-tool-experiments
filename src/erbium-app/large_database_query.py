# imports from langchain community and langchain core packages
from langchain.chains import create_sql_query_chain
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.utilities import SQLDatabase
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


# local imports and python builtins
from config.openai_config import OPENAI_API_KEY
from operator import itemgetter
import os
from typing import List

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


# Replace "sqlite:///MDC.DB" with your actual database URI
db_uri = "sqlite:///erbium-app/MDC.db"

# instantiate the SQLDatabase object
db = SQLDatabase.from_uri(db_uri)


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(..., description="Name of table in SQL database.")


table_names = "\n".join(db.get_usable_table_names())
system1 = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
system2 = f"""Return the names of the SQL tables that are relevant to the user question. \
The tables are:

Music
Business"""


def get_tables(categories: List[Table]) -> List[str]:
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables


# Agent roles are defined and instantiated here
if __name__ == "__main__":
    category_chain = create_extraction_chain_pydantic(
        Table, llm, system_message=system2
    )
    table_chain = category_chain | get_tables
    print(
        table_chain.invoke(
            {"input": "What are all the genres John Doe songs found in album?"}
        )
    )

    query_chain = create_sql_query_chain(llm, db)
    # Convert "question" key to the "input" key expected by current table_chain.
    table_chain = {"input": itemgetter("question")} | table_chain
    # Set table_names_to_use using table_chain.
    full_chain = (
        RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain
    )

    query = full_chain.invoke({"question": "What are all the genres of John Doe songs"})
    print(query)
    print(db.run(query))
