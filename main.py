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

app = FastAPI(title="RAG API Engine v4", lifespan=lifespan)

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
    if not documents: return []
    API_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    pairs = [{"text": query, "text_pair": doc.page_content} for doc in documents]
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": pairs})
        scores = response.json()
        if not isinstance(scores, list) or "error" in scores:
            return [(doc, 0.1) for doc in documents[:top_k]]
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
    except Exception:
        return [(doc, 0.1) for doc in documents[:top_k]]


# --- PRIORITY 8 & 11: BACKGROUND PROCESSING & AUTO-SUMMARIZATION ---
def process_pdf_background(tmp_file_path: str, file_name: str, user_id: str, hf_api_key: str, groq_api_key: str):
    try:
        vector_store = SupabaseVectorStore(
            client=supabase_client, embedding=get_embeddings(hf_api_key),
            table_name="documents", query_name="hybrid_search"
        )
        
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        
        # Priority 11: Generate Title & Summary from first few pages
        preview_text = " ".join([d.page_content for d in docs[:3]])[:4000]
        try:
            summary_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, api_key=groq_api_key)
            sum_prompt = ChatPromptTemplate.from_template("Analyze text and output JSON with keys 'title', 'summary', and 'keywords' (array).\nText: {text}")
            res = (sum_prompt | summary_llm | StrOutputParser()).invoke({"text": preview_text})
            # Clean JSON formatting
            clean_res = res.replace("```json", "").replace("```", "").strip()
            parsed_summary = json.loads(clean_res)
            
            supabase_client.table("document_summaries").insert({
                "user_id": str(user_id),
                "file_name": file_name,
                "title": parsed_summary.get("title", file_name),
                "summary": parsed_summary.get("summary", "No summary available."),
                "keywords": parsed_summary.get("keywords", [])
            }).execute()
        except Exception as e:
            print(f"⚠️ Auto-summary generation skipped: {e}")

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

        del splits
        gc.collect()
            
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
    hf_api_key: str = Form(...),
    groq_api_key: str = Form(...) # Added to support auto-summarization LLM calls
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    fd, tmp_file_path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(await file.read())
        
    background_tasks.add_task(process_pdf_background, tmp_file_path, file.filename, user_id, hf_api_key, groq_api_key)
            
    return {"status": "processing", "message": f"Document '{file.filename}' queued for background processing."}


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
            
            # PRIORITY 12: Intelligent Document Selection (Targeted Scoping)
            user_docs = supabase_client.table("documents").select("metadata").eq("metadata->>user_id", request.user_id).execute()
            available_files = list(set([d["metadata"].get("file_name") for d in user_docs.data if d["metadata"].get("file_name")]))
            
            target_filter = {"user_id": str(request.user_id)}
            if available_files:
                try:
                    selector_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, api_key=request.api_key)
                    sel_prompt = ChatPromptTemplate.from_template("Given the user query and available files, select the single exact file name needed, or output 'ALL'. Available: {files}. Query: {query}")
                    selected_file = (sel_prompt | selector_llm | StrOutputParser()).invoke({"files": json.dumps(available_files), "query": request.query}).strip()
                    if selected_file in available_files:
                        target_filter["file_name"] = selected_file
                except Exception: pass

            # PRIORITY 13: Multi-Query Retrieval Expansion
            queries = [request.query]
            try:
                expander_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2, api_key=request.api_key)
                exp_prompt = ChatPromptTemplate.from_template("Generate 2 alternative variations of this search query for vector retrieval. Return as a comma-separated list.\nQuery: {query}")
                alt_queries = (exp_prompt | expander_llm | StrOutputParser()).invoke({"query": request.query}).split(",")
                queries.extend([q.strip() for q in alt_queries if q.strip()])
            except Exception: pass

            # Aggregate and merge chunks across all query variations
            all_raw_docs = []
            seen_content = set()
            
            for q in queries:
                q_vector = embeddings.embed_query(q)
                hybrid_res = supabase_client.rpc(
                    "hybrid_search",
                    {"query_text": q, "query_embedding": q_vector, "match_count": 5, "filter": target_filter}
                ).execute()
                
                for r in hybrid_res.data:
                    if r["content"] not in seen_content:
                        seen_content.add(r["content"])
                        all_raw_docs.append(Document(page_content=r["content"], metadata={**r["metadata"], "similarity": r["similarity"]}))
            
            best_docs_scored = cross_encode_rerank(request.query, all_raw_docs, request.hf_api_key, top_k=3)
            
            # PRIORITY 10: Confidence Check
            max_confidence = max([score for doc, score in best_docs_scored]) if best_docs_scored else 0
            CONFIDENCE_THRESHOLD = -5.0

            if not best_docs_scored or max_confidence < CONFIDENCE_THRESHOLD:
                yield json.dumps({"type": "metadata", "intent": intent, "sources": []}) + "\n"
                msg = "I could not confidently answer from your uploaded documents."
                for word in msg.split():
                    yield json.dumps({"type": "token", "content": word + " "}) + "\n"
                return
            
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
