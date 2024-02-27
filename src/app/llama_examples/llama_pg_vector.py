import argparse
import os
from pathlib import Path

import psycopg2
from llama_index import ServiceContext
from llama_index import download_loader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import LlamaCPP
from llama_index.node_parser import SentenceSplitter
from llama_index.vector_stores import PGVectorStore
from sqlalchemy import make_url

from src.config.config import OPENAI_API_KEY, DATA_IMPORT_DIRECTORY


class LLamaTest:
    """
    This class contains methods to index and query documents using the LlamaIndex tool.
    """
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
        """Indexes a folder of docx word documents into a Postgresql vector store.

        1. An in ingestion pipeline is created which contains a HuggingFaceEmbedding model along with a
        SentenceSplitter.

        2. It uses LlamaIndex tool DocxReader to load .docx files from
        DATA_IMPORT_DIRECTORY

        3. It runs the ingest pipeline on the list of documents to split and embeds the document texts
        and stores them in the vector store.

        Parameters: None

        Returns: bool: The function returns True signifying the successful completion of the process.
        """

        print("Indexing documents")

        with self.conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
            c.execute(f"CREATE DATABASE {self.db_name}")

        # export the OpenAI API key
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=1024, chunk_overlap=20),
                embed_model,
            ],
            vector_store=self.get_vector_store(),
        )

        docx_reader = download_loader("DocxReader")
        docx_reader = docx_reader()

        documents = []
        # store it for later
        for doc in self.get_docx_filepaths(DATA_IMPORT_DIRECTORY):
            try:
                word_document = docx_reader.load_data(file=doc)
                documents = documents + word_document
            except Exception as e:
                print(f"Error loading document {doc} {e}")
        pipeline.run(documents=documents)

        print("Documents are indexed")
        return True

    def query(self, query: str):
        """Query the vector store for a given query

        Parameters: query (str): The search query

        Returns: response (Response): The response from the query
        """
        print(f"Querying for: {query}")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        vector_store = self.get_vector_store()
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        index = VectorStoreIndex.from_vector_store(
            vector_store, service_context=service_context
        )
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return response

    def get_docx_filepaths(self, directory_path):
        """Load docx files from a particular path

        Parameters: directory_path (str): The path to the directory containing the docx files

        Returns: list: A list of filepaths of the docx files
        """
        filepaths = []

        for filename in os.listdir(directory_path):
            if filename.endswith(".docx"):
                file_path = os.path.join(directory_path, filename)
                filepaths.append(Path(file_path))

        return filepaths

    def get_llm(self) -> LlamaCPP:
        """Create a LlamaCPP LLM object with the model llama-2-13b-chat-GGUF

        Parameters: None

        Returns: llama_index.llms.LlamaCPP: The LlamaCPP object
        """
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
    """This script allows the user to run specific methods of the 'LLamaTest' class by using command-line arguments.

    The user can run either the 'index' or the 'query' method.

    Usage:

    1. To call the 'index' method:
    python script.py --method index

    2. To call the 'query' method:
    python script.py --method query --query "your search query"

    Args:
        --method: Required. Select the method to run, either 'index' or 'query'.
        --query: Required when method= 'query'. Enter the search query.
    """
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
