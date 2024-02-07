from llama_index import ServiceContext, VectorStoreIndex, download_loader, load_index_from_storage, StorageContext
from llama_index.llms import Ollama
from llama_index.embeddings import HuggingFaceEmbedding
from pathlib import Path
import os
from src.open_ai_config.openai_config import PROJECT_FILE_PATH, DATA_PERSIST_DIRECTORY

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class LLamaTestOpen:

    def start(self):

        # export the OpenAI API key
        # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        DocxReader = download_loader("DocxReader")
        docx_reader = DocxReader()
        llm = Ollama(model="llama2", request_timeout=2000)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

        if not os.path.exists(DATA_PERSIST_DIRECTORY):
            # load the documents and create the index
            # documents = SimpleDirectoryReader(DATA_IMPORT_DIRECTORY).load_data()
            documents = []
            # store it for later
            try:
                word_document = docx_reader.load_data(file=Path(PROJECT_FILE_PATH))
                documents = documents + word_document
            except:
                print("Error loading document")
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            index.storage_context.persist(persist_dir=DATA_PERSIST_DIRECTORY)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=DATA_PERSIST_DIRECTORY)
            index = load_index_from_storage(service_context=service_context, storage_context=storage_context)
        query_engine = index.as_query_engine(service_context=service_context)
        response = query_engine.query("What is the Asclepius project about and who is involved in it?")
        return response


if __name__ == "__main__":
    indexer = LLamaTestOpen()
    result = indexer.start()
    print(result)