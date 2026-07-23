import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

def ingest_document():
    # 1. Load the document (Make sure to place a 'sample.pdf' in your folder)
    print("Loading document...")
    loader = PyPDFLoader("sample.pdf")
    docs = loader.load()
    
    # 2. Split the text into manageable chunks
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # 3. Initialize the FastEmbed embedding model
    print("Initializing embedding model...")
    embeddings = FastEmbedEmbeddings() 
    
    # 4. Connect to the Qdrant container and store the vectors
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    print(f"Connecting to Vector DB at {qdrant_url}...")
    
    QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=qdrant_url,
        collection_name="my_documents", 
    )
    print("Ingestion complete! Documents are safely stored in Qdrant.")

if __name__ == "__main__":
    ingest_document()