from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


VECTOR_DB_PATH="data/vectorstore"

def get_retriever(final_top_k=3, initial_fetch_k = 15):

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vectorstore=FAISS.load_local(VECTOR_DB_PATH,embeddings,allow_dangerous_deserialization=True)
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k":initial_fetch_k})

    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    reranker = CrossEncoderReranker(model=cross_encoder_model,top_n=final_top_k)

    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker,base_retriever=base_retriever)

    return compression_retriever
