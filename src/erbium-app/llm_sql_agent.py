# import the necessary packages
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import OpenAI
from open_ai_config.openai_config import OPENAI_API_KEY
import os

# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Replace "sqlite:///MDC.DB" with your actual database URI
db_uri = "sqlite:///erbium-app/MDC.db"

# instantiate the SQLDatabase object
db = SQLDatabase.from_uri(db_uri)


# the language model is instantiated here with OpenAI
llm = OpenAI(temperature=0, verbose=True)

# instantiate the SQLDatabaseChain object
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)


# Agent roles are defined and instantiated here
if __name__ == "__main__":
    db_chain.invoke("how many Developers are in the table?")
