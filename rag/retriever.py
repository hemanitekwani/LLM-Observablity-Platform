from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever , EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever


VECTOR_DB_PATH="data/vectorstore"

def get_retriever(final_top_k=3, initial_fetch_k = 10):

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    vectorstore=FAISS.load_local(VECTOR_DB_PATH,embeddings,allow_dangerous_deserialization=True)
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k":initial_fetch_k})
    
    all_docs = list(vectorstore.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = initial_fetch_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[base_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    reranker = CrossEncoderReranker(model=cross_encoder_model,top_n=final_top_k)

    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker,base_retriever=ensemble_retriever)

    return compression_retriever
