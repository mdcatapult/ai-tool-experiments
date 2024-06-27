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

from llama_index import ServiceContext, VectorStoreIndex, download_loader, load_index_from_storage, StorageContext
from llama_index.llms import Ollama
from llama_index.embeddings import HuggingFaceEmbedding
from pathlib import Path
import os
from src.config.config import PROJECT_FILE_PATH, DATA_PERSIST_DIRECTORY

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class LLamaTestOpen:

    def start(self):
        """Use llama2 and a local file based vector store to index documents and ask a question over them.
        The question is a bit MDC specific and expects that the docs were project PIDs so alter as you see fit.
        """

        # export the OpenAI API key
        # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        DocxReader = download_loader("DocxReader")
        docx_reader = DocxReader()
        llm = Ollama(model="llama2", request_timeout=2000)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

        if not os.path.exists(DATA_PERSIST_DIRECTORY):
            # load the documents and create the index
            documents = []
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