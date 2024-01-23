from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import OpenAI

from open_ai_config.openai_config import OPENAI_API_KEY
import os

# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Replace "sqlite:///MDC.DB" with your actual database URI
db_uri = "sqlite:///MDC.db"
db = SQLDatabase.from_uri(db_uri)


llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)


if __name__ == "__main__":
    db_chain.invoke("How many ids are in the table?")
