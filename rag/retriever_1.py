from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VECTOR_DB_PATH="data/vectorstore"


def get_retriever(final_top_k=3):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    vectorstore=FAISS.load_local(VECTOR_DB_PATH,embeddings,allow_dangerous_deserialization=True)


    base_retriever = vectorstore.as_retriever(search_kwargs={"k":final_top_k})


    return base_retriever



