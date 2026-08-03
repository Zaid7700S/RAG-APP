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
from cachetools import TTLCache

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import time
import fitz

load_dotenv()

supabase_client: Client = None

# Caching for Embeddings & Clients
embedding_cache = TTLCache(maxsize=1000, ttl=3600)
embedding_clients = TTLCache(maxsize=100, ttl=86400) 
upload_statuses = TTLCache(maxsize=1000, ttl=86400) 

SYSTEM_CAPABILITIES_PROMPT = """
You are the intelligent assistant for this Workspace & Document Intelligence Application. 
When users ask what you can do, who you are, or how to use the system, explain your features clearly using these details about THIS application:
1. 📄 **Document RAG & PDF Analysis**: You can upload PDF documents into any session. The backend parses them into text and builds vector embeddings for fast, accurate retrieval.
2. 🌐 **Dual Search Scopes**: Restrict searches to the 'Current Session Only' or search across 'All Uploaded Files' in your database.
3. ⚡ **Smart Routing Modes**: Auto-Router (detects intent automatically), RAG Mode (forces strict document search), and General Mode (direct AI conversation).
4. 🔍 **Hybrid Search & Re-ranking**: Uses vector search combined with Cross-Encoder reranking for high accuracy.
5. 🚀 **Dual Processing Engines**: Supports both Fast Mode (pure text extraction) and Deep Scan (complex layout and table parsing).
Respond in a helpful, friendly, and structured format using clear markdown formatting.
"""

def get_embeddings(hf_api_key: str):
    if not hf_api_key:
        raise HTTPException(status_code=401, detail="Hugging Face API Key is required for embeddings.")
    
    if hf_api_key in embedding_clients:
        return embedding_clients[hf_api_key]
        
    client = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_api_key
    )
    embedding_clients[hf_api_key] = client
    return client

def cached_embed_query(query: str, hf_api_key: str):
    cache_key = f"{hf_api_key[:5]}_{query}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    embeddings_model = get_embeddings(hf_api_key)
    vector = embeddings_model.embed_query(query)
    embedding_cache[cache_key] = vector
    return vector

@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase_client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if supabase_url and supabase_key:
        supabase_client = create_client(supabase_url, supabase_key)
        print("Connected to Supabase successfully!")
    yield

app = FastAPI(title="RAG API Engine v10 - Fast Engine Toggle", lifespan=lifespan)

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
    active_files: list = [] 
    search_all_files: bool = False 

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

def process_pdf_background(tmp_file_path: str, file_name: str, user_id: str, hf_api_key: str, groq_api_key: str, fast_mode: str):
    task_id = f"{user_id}_{file_name}"
    upload_statuses[task_id] = "processing"
    
    try:
        vector_store = SupabaseVectorStore(
            client=supabase_client, embedding=get_embeddings(hf_api_key),
            table_name="documents", query_name="hybrid_search"
        )
        
        doc = fitz.open(tmp_file_path)
        total_pages = len(doc)
        
        # 1. Generate Summary
        preview_text = ""
        for i in range(min(3, total_pages)):
            if fast_mode == "true":
                preview_text += doc[i].get_text("text")
            else:
                preview_text += pymupdf4llm.to_markdown(doc, pages=[i])
        
        try:
            summary_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, api_key=groq_api_key)
            sum_prompt = ChatPromptTemplate.from_template("Analyze text and output JSON with keys 'title', 'summary', and 'keywords' (array).\nText: {text}")
            res = (sum_prompt | summary_llm | StrOutputParser()).invoke({"text": preview_text[:4000]})
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
            print(f"⚠️ Auto-summary skipped: {e}")

        del preview_text
        gc.collect()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""])
        upload_timestamp = datetime.now().isoformat()
        chunk_counter = 1
        
        # --- NEW: Global Chunk Buffer ---
        chunk_buffer = []
        BATCH_SIZE = 20
        
        for page_num in range(total_pages):
            upload_statuses[task_id] = f"processing page {page_num + 1} of {total_pages}"
            
            if fast_mode == "true":
                page_md = doc[page_num].get_text("text")
            else:
                page_md = pymupdf4llm.to_markdown(doc, pages=[page_num])
                
            page_doc = Document(page_content=page_md, metadata={"page": page_num + 1})
            splits = text_splitter.split_documents([page_doc])
            
            for split in splits:
                split.metadata["user_id"] = str(user_id)
                split.metadata["file_name"] = file_name
                split.metadata["chunk_number"] = chunk_counter
                split.metadata["upload_date"] = upload_timestamp
                chunk_counter += 1
            
            # Add this page's chunks to the master buffer
            chunk_buffer.extend(splits)
            
            # --- UPLOAD ONLY WHEN BUFFER IS FULL ---
            while len(chunk_buffer) >= BATCH_SIZE:
                batch = chunk_buffer[:BATCH_SIZE]
                chunk_buffer = chunk_buffer[BATCH_SIZE:] # Remove uploaded chunks from buffer
                
                max_retries = 4
                for attempt in range(max_retries):
                    try:
                        vector_store.add_documents(batch)
                        break 
                    except Exception as e:
                        error_str = str(e).lower()
                        if "429" in error_str or "too many requests" in error_str or attempt < max_retries - 1:
                            sleep_time = (2 ** attempt) * 2 
                            print(f"⚠️ API Rate Limit/Error on Batch. Waiting {sleep_time}s...")
                            time.sleep(sleep_time)
                        else:
                            raise Exception(f"API failed after {max_retries} attempts: {str(e)}")
            
            del page_md
            del page_doc
            del splits
            
            if page_num % 5 == 0 or page_num == total_pages - 1:
                gc.collect()
                
        # --- FLUSH REMAINING CHUNKS ---
        # Upload any leftover chunks after the final page is processed
        if chunk_buffer:
            upload_statuses[task_id] = "Finalizing upload..."
            max_retries = 4
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(chunk_buffer)
                    break 
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "too many requests" in error_str or attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) * 2 
                        time.sleep(sleep_time)
                    else:
                        raise Exception(f"API failed on final batch: {str(e)}")
            chunk_buffer.clear()
        
        doc.close()
        upload_statuses[task_id] = "completed"
        print(f"✅ Background processing complete for {file_name}")
            
    except Exception as e:
        upload_statuses[task_id] = f"failed: {str(e)}"
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
    groq_api_key: str = Form(...),
    fast_mode: str = Form("true") # NEW FIELD
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    MAX_FILE_SIZE = 10 * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    await file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 10MB.")

    fd, tmp_file_path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(await file.read())
        
    background_tasks.add_task(process_pdf_background, tmp_file_path, file.filename, user_id, hf_api_key, groq_api_key, fast_mode)
    return {"status": "processing", "message": f"Document '{file.filename}' queued for parsing."}

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    db_history = supabase_client.table("chat_message_history").select("*").eq("session_id", request.session_id).order("created_at").execute()
    history = []
    history_text_blocks = []
    
    for row in db_history.data:
        if row["role"] == "user": 
            history.append(HumanMessage(content=row["content"]))
            history_text_blocks.append(f"User: {row['content']}")
        else: 
            history.append(AIMessage(content=row["content"]))
            history_text_blocks.append(f"AI: {row['content']}")
    
    try:
        main_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0, api_key=request.api_key)
        fast_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, api_key=request.api_key)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Groq API Key.")

    search_query = request.query
    intent = "GENERAL"
    
    if request.mode.strip() == "RAG": 
        intent = "RAG"
        if history:
            try:
                rewrite_prompt = ChatPromptTemplate.from_template("Rewrite user query to be a standalone search term considering context. History: {history} Query: {query}. Output ONLY the query.")
                search_query = (rewrite_prompt | fast_llm | StrOutputParser()).invoke({"history": "\n".join(history_text_blocks[-4:]), "query": request.query}).strip()
            except: pass
            
    elif request.mode.strip() == "Auto":
        try:
            route_prompt = ChatPromptTemplate.from_template(
                "Analyze the user's latest query given the conversation history.\n"
                "1. Determine intent: 'RAG' (asking about documents/data), 'SYSTEM' (asking how this app works, what you can do, or who you are), or 'GENERAL' (coding, general knowledge).\n"
                "2. Rewrite the query to be standalone if it uses pronouns referring to history. Otherwise, keep it exactly the same.\n\n"
                "History: {history}\nLatest Query: {query}\n\n"
                "Output strictly a JSON object with keys 'intent' and 'search_query'. Do not include markdown blocks."
            )
            res_raw = (route_prompt | fast_llm | StrOutputParser()).invoke({
                "history": "\n".join(history_text_blocks[-4:]) if history else "None", 
                "query": request.query
            }).strip()
            
            clean_json = res_raw.replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(clean_json)
            
            intent = parsed_data.get("intent", "GENERAL").upper()
            search_query = parsed_data.get("search_query", request.query)
        except Exception: 
            pass

    async def generate_chat_stream():
        sources_data = []
        final_answer_accumulator = ""
        
        try:
            if intent == "SYSTEM":
                yield json.dumps({"type": "metadata", "intent": intent, "sources": []}) + "\n"
                messages = [SystemMessage(content=SYSTEM_CAPABILITIES_PROMPT)] + history + [HumanMessage(content=request.query)]
                for chunk in main_llm.stream(messages):
                    token = chunk.content
                    final_answer_accumulator += token
                    yield json.dumps({"type": "token", "content": token}) + "\n"
                    
            elif intent == "RAG":
                target_filter = {"user_id": str(request.user_id)}
                
                if request.search_all_files:
                    user_docs = supabase_client.table("documents").select("metadata").contains("metadata", {"user_id": str(request.user_id)}).execute()
                    available_files = list(set([d["metadata"].get("file_name") for d in user_docs.data if d["metadata"].get("file_name")]))
                    
                    if available_files:
                        try:
                            sel_prompt = ChatPromptTemplate.from_template("Given the query and files, select the exact file name needed, or output 'ALL'. Available: {files}. Query: {query}")
                            selected_file = (sel_prompt | fast_llm | StrOutputParser()).invoke({"files": json.dumps(available_files), "query": search_query}).strip()
                            if selected_file in available_files:
                                target_filter["file_name"] = selected_file
                        except Exception: pass
                else:
                    if request.active_files:
                        if len(request.active_files) == 1:
                            target_filter["file_name"] = request.active_files[0]
                        else:
                            try:
                                sel_prompt = ChatPromptTemplate.from_template("Given the query and files, select the exact file name needed, or output 'ALL'. Available: {files}. Query: {query}")
                                selected_file = (sel_prompt | fast_llm | StrOutputParser()).invoke({"files": json.dumps(request.active_files), "query": search_query}).strip()
                                if selected_file in request.active_files:
                                    target_filter["file_name"] = selected_file
                            except Exception: pass
                    else:
                        yield json.dumps({"type": "metadata", "intent": intent, "sources": []}) + "\n"
                        msg = "There are no documents attached to this session. Please upload a file or switch your scope to 'All Files'."
                        for word in msg.split():
                            yield json.dumps({"type": "token", "content": word + " "}) + "\n"
                        return

                queries = [search_query]
                all_raw_docs = []
                seen_content = set()
                
                for q in queries:
                    q_vector = cached_embed_query(q, request.hf_api_key)
                    hybrid_res = supabase_client.rpc(
                        "hybrid_search",
                        {"query_text": q, "query_embedding": q_vector, "match_count": 15, "filter": target_filter}
                    ).execute()
                    
                    for r in hybrid_res.data:
                        if r["content"] not in seen_content:
                            seen_content.add(r["content"])
                            all_raw_docs.append(Document(page_content=r["content"], metadata={**r["metadata"], "similarity": r["similarity"]}))
                
                best_docs_scored = cross_encode_rerank(search_query, all_raw_docs, request.hf_api_key, top_k=8)
                max_confidence = max([score for doc, score in best_docs_scored]) if best_docs_scored else 0
                CONFIDENCE_THRESHOLD = -8.0

                if not best_docs_scored or max_confidence < CONFIDENCE_THRESHOLD:
                    yield json.dumps({"type": "metadata", "intent": intent, "sources": []}) + "\n"
                    msg = "I could not confidently answer from the selected document scope."
                    for word in msg.split():
                        yield json.dumps({"type": "token", "content": word + " "}) + "\n"
                        final_answer_accumulator += word + " "
                else:
                    for doc, score in best_docs_scored:
                        sources_data.append({
                            "source": doc.metadata.get('file_name', 'Unknown'),
                            "page": doc.metadata.get('page', '?'),
                            "chunk": doc.metadata.get('chunk_number', '?'),
                            "confidence_score": round(score, 3),
                            "snippet": doc.page_content[:150].replace('\n', ' ') + "..."
                        })
                        
                    yield json.dumps({"type": "metadata", "intent": intent, "sources": sources_data}) + "\n"
                    context_text = "\n\n".join([f"Source: {d.metadata.get('file_name')} (Page {d.metadata.get('page')})\nText:\n{d.page_content}" for d, s in best_docs_scored])
                    system_prompt = (
                        "You are an expert document analysis assistant. Answer the user's query using ONLY the provided context below. "
                        "Assume the user is the owner/subject of the documents. "
                        "If the user asks a broad question (like 'what are my marks' or 'summarize'), provide a comprehensive breakdown of all relevant data found in the context. "
                        "If you only have partial data, provide what you have and explicitly state that it is a partial view based on the context. "
                        "If the context contains Markdown tables, numbers, or academic grades, intelligently extract and format them.\n\n"
                        f"Context:\n{context_text}\n\n"
                        "If absolutely NO relevant information can be found in the context to even partially answer the user, output exactly: 'I cannot answer this from the documents.'"
                    )
                    
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
                    
        except Exception as e:
            traceback.print_exc()
            error_message = f"⚠️ System Error during retrieval or generation: {str(e)}"
            yield json.dumps({"type": "metadata", "intent": "ERROR", "sources": []}) + "\n"
            yield json.dumps({"type": "token", "content": error_message}) + "\n"
            final_answer_accumulator = error_message

        finally:
            if final_answer_accumulator:
                supabase_client.table("chat_message_history").insert([
                    {"session_id": request.session_id, "user_id": request.user_id, "role": "user", "content": request.query},
                    {"session_id": request.session_id, "user_id": request.user_id, "role": "ai", "content": final_answer_accumulator}
                ]).execute()

    return StreamingResponse(generate_chat_stream(), media_type="application/x-ndjson")

@app.get("/api/documents/{user_id}")
async def get_user_documents(user_id: str):
    if supabase_client is None: raise HTTPException(status_code=500, detail="Database not configured")
    try:
        summary_res = supabase_client.table("document_summaries").select("*").eq("user_id", user_id).execute()
        summaries = {doc["file_name"]: doc for doc in summary_res.data}
        chunk_res = supabase_client.table("documents").select("metadata").contains("metadata", {"user_id": user_id}).execute()

        final_docs = {}
        for row in chunk_res.data:
            fname = row["metadata"].get("file_name")
            if fname and fname not in final_docs:
                if fname in summaries: final_docs[fname] = summaries[fname]
                else: final_docs[fname] = {"id": fname, "file_name": fname, "title": fname, "summary": "Active in vector knowledge base.", "created_at": row["metadata"].get("upload_date", "Unknown")}
        return {"status": "success", "documents": list(final_docs.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{user_id}/{file_name}")
async def delete_user_document(user_id: str, file_name: str):
    if supabase_client is None: raise HTTPException(status_code=500, detail="Database not configured")
    try:
        supabase_client.table("documents").delete().contains("metadata", {"user_id": user_id, "file_name": file_name}).execute()
        supabase_client.table("document_summaries").delete().eq("user_id", user_id).eq("file_name", file_name).execute()
        return {"status": "success", "message": f"Deleted {file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/sessions/{session_id}")
async def rename_session(session_id: str, request: Request):
    if supabase_client is None: raise HTTPException(status_code=500, detail="Database not configured")
    try:
        body = await request.json()
        new_title = body.get("title")
        if not new_title: raise HTTPException(status_code=400, detail="Title is required")
        supabase_client.table("workspace_sessions").update({"title": new_title}).eq("id", session_id).execute()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/upload/status/")
async def get_upload_status(user_id: str, file_name: str):
    """Allows frontend to poll the real-time background task status."""
    status = upload_statuses.get(f"{user_id}_{file_name}", "unknown")
    return {"status": status}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }
