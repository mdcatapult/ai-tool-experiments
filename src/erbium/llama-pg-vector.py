from langchain_openai import ChatOpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, download_loader, load_index_from_storage, \
    StorageContext, OpenAIEmbedding
from pathlib import Path
import os

from llama_index.extractors import TitleExtractor, SummaryExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import MetadataMode
from llama_index.llms import LlamaCPP
from llama_index.query_engine import RetrieverQueryEngine

from llama_index import SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import PGVectorStore
import textwrap
import openai
import psycopg2
from sqlalchemy import make_url
from llama_index.embeddings import HuggingFaceEmbedding
import argparse

from src.open_ai_config.openai_config import OPENAI_API_KEY, DATA_IMPORT_DIRECTORY
from .vector_db_retriever import VectorDBRetriever


class LLamaTest:

    connection_string = "postgresql://postgres:postgres@localhost:5432"
    db_name = "documents_vector_db"
    conn = psycopg2.connect(connection_string)
    conn.autocommit = True

    url = make_url(connection_string)

    def get_vector_store(self):
        vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=self.url.host,
            password=self.url.password,
            port=self.url.port,
            user=self.url.username,
            table_name="pids",
            embed_dim=384,  # embedding dimension 1536 for openAI, 384 for BAAI
        )
        return vector_store

    def index(self):
        print("Indexing documents")

        with self.conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
            c.execute(f"CREATE DATABASE {self.db_name}")

        # export the OpenAI API key
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        # llm = ChatOpenAI(model_name="gpt-4", temperature=1.0, max_tokens=1000)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=1024, chunk_overlap=20),
                # TitleExtractor(llm=llm),
                # SummaryExtractor(llm=llm),
                embed_model,
            ],
            vector_store=self.get_vector_store(),
        )

        DocxReader = download_loader("DocxReader")
        docx_reader = DocxReader()

        documents = []
        # store it for later
        for doc in self.get_docx_filepaths(DATA_IMPORT_DIRECTORY):
            try:
                word_document = docx_reader.load_data(file=doc)
                documents = documents + word_document
            except:
                print(f"Error loading document {doc}")
        pipeline.run(documents=documents)

        print("Documents are indexed")
        return True

    def query(self, query: str):
        print(f"Querying for: {query}")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # service_context = ServiceContext.from_defaults(
        #     llm=self.getLLM(), embed_model=embed_model
        # )
        vector_store = self.get_vector_store()
        # retriever = VectorDBRetriever(
        #     vector_store, embed_model, query_mode="default", similarity_top_k=2
        # )
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        # query_engine = RetrieverQueryEngine.from_args(
        #     retriever, service_context=service_context
        # )
        index = VectorStoreIndex.from_vector_store(
            vector_store, service_context=service_context
        )
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return response

    def get_docx_filepaths(self, directory_path):
        filepaths = []

        for filename in os.listdir(directory_path):
            if filename.endswith(".docx"):
                file_path = os.path.join(directory_path, filename)
                filepaths.append(Path(file_path))

        return filepaths

    def getLLM(self) -> LlamaCPP:
        model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
        llm = LlamaCPP(
            # You can pass in the URL to a GGML model to download it automatically
            model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=None,
            temperature=0.1,
            max_new_tokens=256,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=3900,
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": 1},
            verbose=True,
        )
        return llm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run methods in LangChainAgent")
    parser.add_argument(
    "--method", choices=["index", "query"], help="Select the method to run", required=True
    )
    parser.add_argument(
        "--query", type=str, help="Enter the query to search", required=False
    )

    llama_test = LLamaTest()
    args = parser.parse_args()
    if args.method == "index":
        result = llama_test.index()
    elif args.method == "query":
        result = llama_test.query(args.query)
    else:
        result = "Invalid method specified!"
    print(result)
