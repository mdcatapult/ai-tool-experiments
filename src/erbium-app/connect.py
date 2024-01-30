# imports from langchain community and langchain core packages
from langchain.chains import create_sql_query_chain
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.utilities import SQLDatabase
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


# local imports and python builtins
from config.openai_config import (
    OPENAI_API_KEY,
    DATABASE_NAME,
    DATABASE_PASS,
    DATABASE_USER,
    DATABASE_HOST,
    DATABASE_PORT,
    DATABASE_SCHEMA_NAME,
)
from operator import itemgetter
import os
import psycopg2
from sqlalchemy import create_engine
from typing import List


# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# postgresql database connection parameters
db_uri = f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
engine = create_engine(
    db_uri,
    connect_args={"options": "-csearch_path={}".format(DATABASE_SCHEMA_NAME)},
)


# chatopenai language model
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# instantiate the SQLDatabase object
db = SQLDatabase(engine=engine)


# connect to the PostgreSQL database server
class ConnectPSQLDatabase:
    def connect(self, config):
        """Connect to the PostgreSQL database server"""
        try:
            # connecting to the PostgreSQL server
            with psycopg2.connect(**config) as conn:
                print("Connected to the PostgreSQL server.")

                cursor = conn.cursor()
                query = "SELECT * FROM coshh.chemical;"
                cursor.execute(query)
                # Fetch all rows
                rows = cursor.fetchall()

                # Iterate over the rows and print the results
                for row in rows:
                    print(row)

            cursor.close()
            conn.close()
        except (psycopg2.DatabaseError, Exception) as error:
            print(error)


# create a Table class that inherits from BaseModel
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

chemical
audit"""


def get_tables(categories: List[Table]) -> List[str]:
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


if __name__ == "__main__":
    category_chain = create_extraction_chain_pydantic(
        Table, llm, system_message=system2
    )
    table_chain = category_chain | get_tables
    print(table_chain.invoke({"input": "the tables in the schema coshh?"}))

    query_chain = create_sql_query_chain(llm, db)
    # Convert "question" key to the "input" key expected by current table_chain.
    table_chain = {"input": itemgetter("question")} | table_chain
    # Set table_names_to_use using table_chain.
    full_chain = (
        RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain
    )

    query = full_chain.invoke({"question": "which chemical is located in lab 4?"})
    print(query)
    print(db.run(query))

# How many chemical are in the chemical table
