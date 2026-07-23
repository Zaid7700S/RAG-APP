import os
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

load_dotenv()

def query_rag():
    # 1. Connect to Qdrant and set up the Retriever
    embeddings = FastEmbedEmbeddings()
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    
    client = QdrantClient(url=qdrant_url)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_documents",
        embedding=embeddings, 
    )
    
    # Retrieve the top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Initialize the Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1
    )

    # 3. Set up the Prompt Template
    template = """Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    
    Context: {context}
    
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)

    # 4. Helper function to format the retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Build the LCEL RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Execute a sample query
    question = "What is the main topic of the document?"
    print(f"Question: {question}")
    
    print("\nThinking...")
    result = rag_chain.invoke(question)
    
    print("\nAnswer:")
    print(result)

if __name__ == "__main__":
    query_rag()