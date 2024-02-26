import argparse

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from src.config.config import OPENAI_API_KEY

import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class LangchainAgent:

    def retrieval(self):
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
        docs = loader.load()
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)

        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        
        Question: {input}""")

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
        return response["answer"]

    def conversation(self):
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
        docs = loader.load()
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        retriever = vector.as_retriever()
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        retriever_chain.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, prompt)

        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })
        return response["answer"]

    def agent(self):
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
        docs = loader.load()
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        retriever = vector.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",

        )
        # Use duckduckgo to assist the model
        search_tool = DuckDuckGoSearchRun()
        tools = [retriever_tool, search_tool]
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        agent_executor.invoke({"input": "how can langsmith help with testing?"})
        agent_executor.invoke({"input": "what is the weather in Nether Alderley, UK?"})
        # This next one seems to go a bit off piste and talk about a different langsmith than the one expected.
        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        agent_executor.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run methods in LangChainAgent")
    parser.add_argument(
        "method", choices=["retrieval", "conversation", "agent"], help="Select the method to run"
    )

    langchain_test = LangchainAgent()
    args = parser.parse_args()
    if args.method == "retrieval":
        result = langchain_test.retrieval()
    elif args.method == "conversation":
        result = langchain_test.conversation()
    elif args.method == "agent":
        result = langchain_test.agent()
    else:
        result = "Invalid method specified!"
    print(result)
