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

# import the necessary packages
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import OpenAI

from src.config.config import OPENAI_API_KEY
import os

# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Replace "sqlite:///MDC.DB" with your actual database URI. Note that '///' is a relative path and '////' is an absolute path
db_uri = "sqlite:///data/MDC.db"

# instantiate the SQLDatabase object
db = SQLDatabase.from_uri(db_uri)


# the language model is instantiated here with OpenAI
llm = OpenAI(temperature=0, verbose=True)

# instantiate the SQLDatabaseChain object
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)


# Agent roles are defined and instantiated here
if __name__ == "__main__":
    db_chain.invoke("how many Developers are in the table?")
