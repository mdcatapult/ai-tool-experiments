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
from langchain.chains import create_sql_query_chain
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.utilities import SQLDatabase
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

import pandas as pd

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

"""
system2 = f"""Return the names of the SQL tables that MIGHT be relevant and irrelevant to the user question. \
The tables are:
{table_names}
audit
"""


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


def get_non_alphanumeric_input():
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


def main():
    category_chain = create_extraction_chain_pydantic(
        Table, llm, system_message=system2
    )
    table_chain = category_chain | get_tables
    print(table_chain.invoke({"input": "the tables in the schema coshh?"}))

    # Create query chain.
    query_chain = create_sql_query_chain(llm, db)
    # Convert "question" key to the "input" key expected by current table_chain.

    table_chain = {"input": itemgetter("question")} | table_chain
    # Set table_names_to_use using table_chain.
    full_chain = (
        RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain
    )
    query = full_chain.invoke({"question": get_non_alphanumeric_input()})
    print(query)
    print(pd.read_sql(query, engine).to_string())


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
# 18                 id
# 19         cas_number
# 20      chemical_name
# 21    chemical_number
# 22       matter_state
# 23           quantity
# 24              added
# 25             expiry
# 26  safety_data_sheet
# 27         coshh_link
# 28       lab_location
# 29           cupboard
# 30   project_specific
# 31       storage_temp
# 32        is_archived
# 33     chemical_owner
# 34    last_updated_by
