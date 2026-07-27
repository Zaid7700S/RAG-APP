import os
import tempfile
import traceback
import gc
import json
import requests
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# --- STATE MANAGEMENT & LAZY LOADING ---
supabase_client: Client = None

def get_embeddings(hf_api_key: str):
    if not hf_api_key:
        raise HTTPException(status_code=401, detail="Hugging Face API Key is required for embeddings.")
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_api_key
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase_client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if supabase_url and supabase_key:
        supabase_client = create_client(supabase_url, supabase_key)
        print("Connected to Supabase successfully!")
    yield

app = FastAPI(title="RAG API Engine v3", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"CRASH LOG: {traceback.format_exc()}")
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500, content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": origin, "Access-Control-Allow-Credentials": "true"}
    )

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    api_key: str          
    hf_api_key: str
    mode: str = "Auto"    


def cross_encode_rerank(query: str, documents: list[Document], hf_api_key: str, top_k: int = 3):
    """Uses Hugging Face API to rerank chunks and returns a tuple of (Document, Score)."""
    if not documents: return []
    
    API_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    pairs = [{"text": query, "text_pair": doc.page_content} for doc in documents]
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": pairs})
        scores = response.json()
        
        if not isinstance(scores, list) or "error" in scores:
            # Fallback pseudo-scores if API is cold
            return [(doc, float(doc.metadata.get("similarity", 0.1))) for doc in documents[:top_k]]
            
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
    except Exception:
        return [(doc, 0.1) for doc in documents[:top_k]]


# --- PRIORITY 8: BACKGROUND PROCESSING WORKER ---
def process_pdf_background(tmp_file_path: str, file_name: str, user_id: str, hf_api_key: str):
    try:
        vector_store = SupabaseVectorStore(
            client=supabase_client, embedding=get_embeddings(hf_api_key),
            table_name="documents", query_name="hybrid_search"
        )
        
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])
        splits = text_splitter.split_documents(docs)
        del docs
        gc.collect()

        upload_timestamp = datetime.now().isoformat()
        for idx, split in enumerate(splits):
            split.metadata["user_id"] = str(user_id)
            split.metadata["file_name"] = file_name
            split.metadata["chunk_number"] = idx + 1
            split.metadata["upload_date"] = upload_timestamp
        
        BATCH_SIZE = 15  
        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i : i + BATCH_SIZE]
            vector_store.add_documents(batch)
            del batch
            gc.collect()
            
        print(f"✅ Background processing complete for {file_name}")
    except Exception as e:
        print(f"❌ Background processing failed for {file_name}: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.post("/upload/")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    user_id: str = Form(...),
    hf_api_key: str = Form(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Save to a persistent temporary file before passing to background worker
    fd, tmp_file_path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(await file.read())
        
    # Send heavy lifting to background
    background_tasks.add_task(process_pdf_background, tmp_file_path, file.filename, user_id, hf_api_key)
            
    return {"status": "processing", "message": f"Document '{file.filename}' is being processed in the background."}


@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    db_history = supabase_client.table("chat_message_history").select("*").eq("session_id", request.session_id).order("created_at").execute()
    history = []
    for row in db_history.data:
        if row["role"] == "user": history.append(HumanMessage(content=row["content"]))
        else: history.append(AIMessage(content=row["content"]))
    
    try:
        main_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0, api_key=request.api_key)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Groq API Key.")

    intent = "GENERAL"
    if request.mode.strip() == "RAG": intent = "RAG"
    elif request.mode.strip() == "Auto":
        try:
            router = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, max_tokens=5, api_key=request.api_key)
            route_prompt = ChatPromptTemplate.from_template("Classify user input into 'RAG' (documents/files/summarize) or 'GENERAL' (trivia/general knowledge/greetings). Input: {input}. Output ONE WORD.")
            intent_raw = (route_prompt | router | StrOutputParser()).invoke({"input": request.query}).strip().upper()
            if "RAG" in intent_raw: intent = "RAG"
        except Exception: pass

    async def generate_chat_stream():
        sources_data = []
        final_answer_accumulator = ""
        
        if intent == "RAG":
            embeddings = get_embeddings(request.hf_api_key)
            query_vector = embeddings.embed_query(request.query)
            
            hybrid_res = supabase_client.rpc(
                "hybrid_search",
                {"query_text": request.query, "query_embedding": query_vector, "match_count": 10, "filter": {"user_id": str(request.user_id)}}
            ).execute()
            
            # Carry over hybrid similarity score into metadata
            raw_docs = [Document(page_content=r["content"], metadata={**r["metadata"], "similarity": r["similarity"]}) for r in hybrid_res.data]
            
            best_docs_scored = cross_encode_rerank(request.query, raw_docs, request.hf_api_key, top_k=3)
            
            # PRIORITY 10: Confidence Threshold Check
            max_confidence = max([score for doc, score in best_docs_scored]) if best_docs_scored else 0
            CONFIDENCE_THRESHOLD = -5.0 # Cross-encoders can output negative logits, adjust this based on testing

            if not best_docs_scored or max_confidence < CONFIDENCE_THRESHOLD:
                yield json.dumps({"type": "metadata", "intent": intent, "sources": []}) + "\n"
                msg = "I could not confidently answer from your uploaded documents (Confidence score too low). Please rephrase or upload a relevant document."
                for word in msg.split():
                    yield json.dumps({"type": "token", "content": word + " "}) + "\n"
                return
            
            # PRIORITY 9: Rich Citations Integration
            for doc, score in best_docs_scored:
                sources_data.append({
                    "source": doc.metadata.get('file_name', 'Unknown'),
                    "page": doc.metadata.get('page', '?'),
                    "chunk": doc.metadata.get('chunk_number', '?'),
                    "confidence_score": round(score, 3),
                    "snippet": doc.page_content[:150].replace('\n', ' ') + "..."
                })
                
            yield json.dumps({"type": "metadata", "intent": intent, "sources": sources_data}) + "\n"
            
            context_text = "\n\n".join([f"Source: {d.metadata.get('file_name')} (Page {d.metadata.get('page')})\nText: {d.page_content}" for d, s in best_docs_scored])
            system_prompt = f"You are a strict document assistant. Answer ONLY using this context:\n{context_text}\nIf not in context, say 'I cannot answer this from the documents.'"
            
            messages = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=request.query)]
            
            for chunk in main_llm.stream(messages):
                token = chunk.content
                final_answer_accumulator += token
                yield json.dumps({"type": "token", "content": token}) + "\n"
                
        else:
            yield json.dumps({"type": "metadata", "intent": intent, "sources": []}) + "\n"
            messages = [SystemMessage(content="You are a helpful AI assistant.")] + history + [HumanMessage(content=request.query)]
            
            for chunk in main_llm.stream(messages):
                token = chunk.content
                final_answer_accumulator += token
                yield json.dumps({"type": "token", "content": token}) + "\n"

        supabase_client.table("chat_message_history").insert([
            {"session_id": request.session_id, "user_id": request.user_id, "role": "user", "content": request.query},
            {"session_id": request.session_id, "user_id": request.user_id, "role": "ai", "content": final_answer_accumulator}
        ]).execute()

    return StreamingResponse(generate_chat_stream(), media_type="application/x-ndjson")


@app.get("/db/explore/")
async def explore_database(limit: int = 5):
    if supabase_client is None: return {"status": "error", "message": "Supabase not configured"}
    response = supabase_client.table("documents").select("id, content, metadata").limit(limit).execute()
    return {"status": "success", "data": response.data}
