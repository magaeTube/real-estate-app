# from app.core.config import get_config_info
from langchain_chroma import Chroma
import chromadb

def connect_chromadb(site: str):
    host = site["host"]
    port = site["port"]
    print(host, port)
    client = chromadb.HttpClient(host=host, port=int(port))

    return client


def get_chromadb(site: str, collection_name: str, embeddings_model):
    client = connect_chromadb(site)
    db = Chroma(client=client,
                collection_name=collection_name,
                embedding_function=embeddings_model)

    return db
