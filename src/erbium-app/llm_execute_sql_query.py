# import the necessary packages fron langchain community and langchain core packages
# python -m langchain_community.tools.sql_database.tool --help
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_openai import ChatOpenAI

# local imports and python builtins
from config.openai_config import OPENAI_API_KEY
from operator import itemgetter
import os

# import the OpenAI API key from the os environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Replace "sqlite:///MDC.DB" with your actual database URI
db_uri = "sqlite:///erbium-app/MDC.db"

# instantiate the SQLDatabase object
db = SQLDatabase.from_uri(db_uri)

# the language model is instantiated here with OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# execute the SQL query and return the result
execute_query = QuerySQLDataBaseTool(db=db, verbose=True)
write_query = create_sql_query_chain(llm=llm, db=db)

# instantiate the SQLDatabaseChain object
chain = write_query | execute_query

# instantiate the PromptTemplate object which gives the answer
answer_prompt = PromptTemplate.from_template(
    """ Given the following user question, corresponding SQL query, and SQL result, answer the user question.
  
  
  Question: {question}
  SQL Query: {query}
  SQL Result: {result}

  Answer: """
)

llm_response = answer_prompt | llm | StrOutputParser()


# instantiate the Runnable object that will be used to run the SQL query
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | llm_response
)


# Agent roles are defined and instantiated here
if __name__ == "__main__":
    chain.get_prompts()[0].pretty_print()
    print(chain.invoke({"question": "How many employees are there?"}))
