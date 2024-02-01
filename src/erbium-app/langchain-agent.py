from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from src.open_ai_config.openai_config import OPENAI_API_KEY


class LangchainAgent:

    def start(self):
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

if __name__ == "__main__":
    langchain_test = LangchainAgent()
    result = langchain_test.start()
    print(result)
