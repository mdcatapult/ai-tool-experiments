from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import argparse

from src.open_ai_config.openai_config import OPENAI_API_KEY


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run methods in LangChainAgent")
    parser.add_argument(
        "method", choices=["retrieval", "conversation"], help="Select the method to run"
    )

    langchain_test = LangchainAgent()
    args = parser.parse_args()
    if args.method == "retrieval":
        result = langchain_test.retrieval()
    elif args.method == "conversation":
        result = langchain_test.conversation()
    else:
        result = "Invalid method specified!"
    print(result)
