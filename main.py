import os
import tempfile
import traceback
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# --- FIX: OVERRIDE BROKEN LANGCHAIN SUPABASE SEARCH METHODS ---
def _patched_similarity_search_by_vector(self, embedding: list, k: int = 4, filter: dict = None, **kwargs):
    client = getattr(self, "client", None) or getattr(self, "_client", None)
    query_name = getattr(self, "query_name", "match_documents")
    
    if not client:
        raise ValueError("SupabaseVectorStore missing client connection.")

    match_params = {"query_embedding": embedding, "match_count": k}
    if filter:
        match_params["filter"] = filter
        
    res = client.rpc(query_name, match_params).execute()
    
    return [
        Document(
            page_content=record.get("content", ""),
            metadata=record.get("metadata", {})
        ) 
        for record in res.data
    ]

def _patched_similarity_search(self, query: str, k: int = 4, filter: dict = None, **kwargs):
    embedding_model = getattr(self, "embedding", None) or getattr(self, "_embedding", None)
    if not embedding_model:
        raise ValueError("SupabaseVectorStore missing embedding model.")
    query_embedding = embedding_model.embed_query(query)
    return self.similarity_search_by_vector(query_embedding, k=k, filter=filter, **kwargs)

# Inject both patches into SupabaseVectorStore
SupabaseVectorStore.similarity_search_by_vector = _patched_similarity_search_by_vector
SupabaseVectorStore.similarity_search = _patched_similarity_search


# --- STATE MANAGEMENT & LAZY LOADING ---
chat_sessions = {}
supabase_client: Client = None

def get_embeddings(google_api_key: str):
    if not google_api_key:
        raise HTTPException(status_code=401, detail="Google API Key is required for embeddings.")
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=google_api_key
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google API Key: {str(e)}")


def get_vector_store(google_api_key: str):
    global supabase_client
    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
    return SupabaseVectorStore(
        client=supabase_client,
        embedding=get_embeddings(google_api_key),
        table_name="documents",
        query_name="match_documents"
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase_client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if supabase_url and supabase_key:
        supabase_client = create_client(supabase_url, supabase_key)
        print("Connected to Supabase successfully!")
    else:
        print("WARNING: Supabase credentials not found in .env")
    yield

app = FastAPI(title="RAG API Engine", lifespan=lifespan)


# --- GLOBAL EXCEPTION HANDLER ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"CRASH LOG: {traceback.format_exc()}")
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": origin, 
            "Access-Control-Allow-Credentials": "true"
        }
    )


# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- MODELS ---
class ChatRequest(BaseModel):
    session_id: str
    query: str
    api_key: str          
    google_api_key: str
    mode: str = "Auto"    

class ChatResponse(BaseModel):
    intent: str
    answer: str
    sources: list = []


# --- ENDPOINTS ---

@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...), 
    user_id: str = Form(...),
    google_api_key: str = Form(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    if not user_id or user_id == "undefined":
        raise HTTPException(status_code=401, detail="Invalid user session. Please log in again.")
    
    if not google_api_key:
        raise HTTPException(status_code=401, detail="Google API Key is required to process documents.")

    print(f"📥 UPLOADING FILE: '{file.filename}' FOR USER: {user_id}")

    vector_store = get_vector_store(google_api_key)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_path = os.path.join(tmp_dir, file.filename)
        with open(tmp_file_path, "wb") as f:
            f.write(await file.read())
        
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Free memory from original document objects
        del docs
        gc.collect()

        # Attach user_id and file_name to metadata for user isolation
        for split in splits:
            split.metadata["user_id"] = str(user_id)
            split.metadata["file_name"] = file.filename
        
        # --- BATCH PROCESSING LOGIC (Prevents 512MB RAM Overflows) ---
        BATCH_SIZE = 15  
        total_chunks = len(splits)
        print(f"📦 Total Chunks: {total_chunks}. Processing in batches of {BATCH_SIZE}...")

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = splits[i : i + BATCH_SIZE]
            print(f"⏳ Embedding batch {i // BATCH_SIZE + 1} of {(total_chunks + BATCH_SIZE - 1) // BATCH_SIZE}...")
            
            vector_store.add_documents(batch)
            
            # Explicitly free batch memory after write
            del batch
            gc.collect()

        # Final cleanup
        del splits
        gc.collect()
            
    return {"status": "success", "message": f"Document '{file.filename}' successfully processed and stored for user {user_id}."}


@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.session_id:
        request.session_id = "default"
        
    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = []
    
    history = chat_sessions[request.session_id]
    
    try:
        main_llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            temperature=0.0, 
            api_key=request.api_key
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Groq API Key provided.")
    
    vector_store = get_vector_store(request.google_api_key)

    # --- SMART INTENT ROUTING ---
    selected_mode = request.mode.strip()
    if selected_mode == "General":
        intent = "GENERAL"
    elif selected_mode == "RAG":
        intent = "RAG"
    else:
        try:
            router_llm = ChatGroq(
                model_name="llama-3.1-8b-instant", 
                temperature=0.0, 
                max_tokens=5,
                api_key=request.api_key
            )
            classifier_prompt = ChatPromptTemplate.from_template(
                """You are an intelligent routing agent. Classify the user's input into exactly one of two categories: 'RAG' or 'GENERAL'.
                
                User Input: "{input}"
                
                When to choose RAG:
                - The user mentions "document", "file", "pdf", or "upload".
                - The user asks to "summarize", "analyze", or "explain" a provided text.
                - The user asks a question referring to a provided context using words like "this" or "it" (e.g., "what is this about?").
                
                When to choose GENERAL:
                - The user asks general knowledge, trivia, definitions, or coding questions (e.g., "What is Groq?", "What is the capital of France?").
                - The user says hello or makes casual conversation.
                - If the question is a standalone fact that does not explicitly reference a document, ALWAYS default to GENERAL.
                
                Output ONLY ONE WORD: RAG or GENERAL."""
            )
            classifier_chain = classifier_prompt | router_llm | StrOutputParser()
            intent_raw = classifier_chain.invoke({"input": request.query}).strip().upper()
            
            # Smart Fallback: Default to GENERAL unless the model explicitly chose RAG
            if "RAG" in intent_raw and "GENERAL" not in intent_raw:
                intent = "RAG"
            else:
                intent = "GENERAL"
                
        except Exception as e:
            print(f"Router error: {e}, falling back to GENERAL")
            intent = "GENERAL"

    print(f"🎯 ROUTED INTENT: {intent} (Mode chosen: {selected_mode})")

    # Execution
    sources_data = []
    try:
        if intent == "GENERAL":
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Answer naturally based on the conversation history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            general_chain = general_prompt | main_llm | StrOutputParser()
            final_answer = general_chain.invoke({"input": request.query, "chat_history": history})
            
        else:
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 3, 
                    "filter": {"user_id": str(request.session_id)}
                }
            )
            
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(main_llm, retriever, contextualize_prompt)

            # --- STRICT RAG ISOLATION PROMPT ---
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a strict document retrieval assistant. 
                You must answer the question ONLY using the exact context provided below. 
                
                CRITICAL INSTRUCTION: If the context is completely empty, or if the answer cannot be found in the context, you MUST reply exactly with: "I cannot answer this based on the provided documents in this session."
                DO NOT use your pre-trained general knowledge. DO NOT guess.
                
                Context:
                {context}"""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(main_llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": request.query, "chat_history": history})
            final_answer = response["answer"]
            
            for doc in response.get("context", []):
                sources_data.append({
                    "source": os.path.basename(doc.metadata.get('source', doc.metadata.get('file_name', 'Unknown'))),
                    "snippet": doc.page_content[:200].replace('\n', ' ')
                })
    except Exception as e:
        traceback.print_exc()
        final_answer = f"⚠️ AI Generation Error: {str(e)}"

    # Save to Memory
    history.append(HumanMessage(content=request.query))
    history.append(AIMessage(content=final_answer))
    
    return ChatResponse(intent=intent, answer=final_answer, sources=sources_data)


@app.get("/db/explore/")
async def explore_database(limit: int = 5):
    global supabase_client
    if supabase_client is None:
        return {"status": "error", "message": "Supabase not configured"}
        
    response = supabase_client.table("documents").select("id, content, metadata").limit(limit).execute()
    
    db_visual = []
    for record in response.data:
        db_visual.append({
            "id": record["id"],
            "metadata": record["metadata"],
            "text_content": record["content"]
        })
        
    return {"status": "success", "total_previewed": len(db_visual), "data": db_visual}


@app.delete("/db/clear/")
async def clear_database():
    global supabase_client
    if supabase_client is None:
        return {"status": "error"}
    
    supabase_client.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    return {"status": "success", "message": "Supabase vector table wiped clean."}
