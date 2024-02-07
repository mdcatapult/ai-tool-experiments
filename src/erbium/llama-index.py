from llama_index import VectorStoreIndex, SimpleDirectoryReader, download_loader, load_index_from_storage, StorageContext
from pathlib import Path
import os
from src.open_ai_config.openai_config import OPENAI_API_KEY, DATA_IMPORT_DIRECTORY, DATA_PERSIST_DIRECTORY


class LLamaTest:

    def start(self):

        # export the OpenAI API key
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        DocxReader = download_loader("DocxReader")
        docx_reader = DocxReader()

        if not os.path.exists(DATA_PERSIST_DIRECTORY):
            # load the documents and create the index
            # documents = SimpleDirectoryReader(DATA_IMPORT_DIRECTORY).load_data()
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
        response = query_engine.query("What is the Yttrium project about?")
        return response

    def get_docx_filepaths(self, directory_path):
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