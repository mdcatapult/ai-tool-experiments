from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os
from src.open_ai_config.openai_config import OPENAI_API_KEY, DATA_IMPORT_DIRECTORY, DATA_PERSIST_DIRECTORY

# export the OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not os.path.exists(DATA_PERSIST_DIRECTORY):
    # load the documents and create the index
    documents = SimpleDirectoryReader(DATA_IMPORT_DIRECTORY).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=DATA_PERSIST_DIRECTORY)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=DATA_PERSIST_DIRECTORY)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

