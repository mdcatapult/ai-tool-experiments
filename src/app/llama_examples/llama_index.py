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

import os
from pathlib import Path

from llama_index import VectorStoreIndex, download_loader, load_index_from_storage, StorageContext

from src.config.config import OPENAI_API_KEY, DATA_IMPORT_DIRECTORY, DATA_PERSIST_DIRECTORY


class LLamaTest:

    def start(self):
        """Load files into vector store. Asks a simple question at the end of the process"""

        # export the OpenAI API key
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        DocxReader = download_loader("DocxReader")
        docx_reader = DocxReader()

        if not os.path.exists(DATA_PERSIST_DIRECTORY):
            documents = []
            # store it for later
            for doc in self.get_docx_filepaths(DATA_IMPORT_DIRECTORY):
                try:
                    word_document = docx_reader.load_data(file=doc)
                    documents = documents + word_document
                except:
                    print("Error loading document")
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=DATA_PERSIST_DIRECTORY)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=DATA_PERSIST_DIRECTORY)
            index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query("What is the Asclepius project about and who is involved in it?")
        return response

    def get_docx_filepaths(self, directory_path):
        """Load docx files from a particular path"""

        filepaths = []

        for filename in os.listdir(directory_path):
            if filename.endswith(".docx"):
                file_path = os.path.join(directory_path, filename)
                filepaths.append(Path(file_path))

        return filepaths

if __name__ == "__main__":
    indexer = LLamaTest()
    result = indexer.start()
    print(result)