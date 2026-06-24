from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path



PDF_PATH="data/attention-is-all-you-need.pdf"

VECTOR_DB_PATH="data/vectorstore"


def create_vector_store():
    print("Loading PDF..")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print(f"Loaded: {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=150)

    chunks=splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print("Creating FAISS index")

    vector_store = FAISS.from_documents(chunks,embeddings)

    Path(VECTOR_DB_PATH).mkdir(parents=True,exist_ok=True)

    vector_store.save_local(VECTOR_DB_PATH)

    print("Vectorstore saved.")



if __name__ == "__main__":
    create_vector_store()








