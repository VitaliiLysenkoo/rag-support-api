from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import os
import glob

from app.database import get_db
from app.repository import DocumentRepository
from app.rag_engine import process_and_store_document, search_similar_chunks
from app.ai_service import get_chat_response, get_embedding
from app.config import setup_logger

logger = setup_logger("router_rag")

router = APIRouter(
    tags=["RAG Core"]
)

class QueryRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Who is my manager at Valve according to the handbook?"
                }
            ]
        }
    }

# Global set to track files currently being processed in background
processing_files = set()

@router.post("/seed", summary="Seed database with documents")
async def seed_documents(background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    """
    Scans the `test_documents` directory and queues all TXT and PDF files for semantic chunking and vectorization.
    Old documents in the database are selectively cleared to prevent duplicates.
    """
    # Resolve absolute path to the test_documents directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doc_dir = os.path.join(base_dir, "test_documents")
    
    files_to_process = []
    files_to_process.extend(glob.glob(os.path.join(doc_dir, "*.txt")))
    files_to_process.extend(glob.glob(os.path.join(doc_dir, "*.pdf")))
    
    if not files_to_process:
        return {"message": "No .txt or .pdf files found in test_documents folder"}

    existing_docs = await DocumentRepository.get_all_document_names(db)
    
    queued_count = 0
    skipped_count = 0

    for file_path in files_to_process:
        doc_name = os.path.basename(file_path)
        
        # Prevent duplicate processing if the file is already in DB or currently in the background queue
        if doc_name in existing_docs or doc_name in processing_files:
            skipped_count += 1
            continue
            
        processing_files.add(doc_name)
        background_tasks.add_task(process_and_store_document, file_path)
        queued_count += 1
        
    if queued_count == 0:
        return {"message": f"All {skipped_count} files are already imported or currently processing."}
        
    # Clear semantic cache because underlying documents are changing
    await DocumentRepository.clear_semantic_cache(db)
    logger.info("Semantic cache cleared due to new document ingestion.")
        
    return {"message": f"Started processing {queued_count} new documents. Skipped {skipped_count} already imported/queued."}

@router.post("/ask", summary="Ask a question using RAG")
async def ask_question(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Core RAG endpoint. Searches for context in the PostgreSQL vector database using L2 distance, 
    and generates an accurate answer via OpenAI's LLM, citing specific document pages.
    """
    logger.info(f"Received query: '{request.question}'")
    
    # 1. Embed the question for semantic cache and search
    query_vector = await get_embedding(request.question)
    
    # 2. Check semantic cache
    cached_result = await DocumentRepository.check_semantic_cache(db, query_vector, max_distance=0.05)
    if cached_result:
        logger.info("Semantic cache HIT! Returning instant answer.")
        return {
            "answer": cached_result.answer,
            "sources": cached_result.sources,
            "cached": True
        }
        
    logger.info("Semantic cache MISS. Proceeding with RAG.")
    
    # We still use the original string query for search_similar_chunks in this version,
    # or we could refactor search_similar_chunks. For simplicity, we just pass request.question.
    similar_chunks = await search_similar_chunks(request.question, db, limit=30)
    
    if not similar_chunks:
        logger.warning("No similar chunks found (database empty).")
        return {"answer": "No relevant documents found in the database. Please run /seed first and wait a minute for the background process to finish.", "sources": []}

    logger.info(f"Retrieved {len(similar_chunks)} chunks for context.")

    # Format sources with IDs so the LLM can cite them
    context_text = ""
    for i, chunk in enumerate(similar_chunks):
        doc_info = chunk.document_name
        if getattr(chunk, 'page_number', None) is not None:
            doc_info += f" (Page {chunk.page_number})"
        context_text += f"[SOURCE ID: {i}] ({doc_info})\n{chunk.content}\n\n---\n\n"
    
    ai_response = await get_chat_response(request.question, context_text)
    
    answer = ai_response.get("answer", "")
    used_ids = ai_response.get("used_source_ids", [])
    
    # Filter sources to only include the ones the LLM actually used
    sources = []
    for i in used_ids:
        if 0 <= i < len(similar_chunks):
            chunk = similar_chunks[i]
            doc_info = chunk.document_name
            if getattr(chunk, 'page_number', None) is not None:
                doc_info += f" (Page {chunk.page_number})"
                
            sources.append({
                "document": doc_info,
                "snippet": chunk.content[:100] + "..."
            })
            
    # 3. Save the new answer to semantic cache
    await DocumentRepository.save_to_semantic_cache(db, request.question, query_vector, answer, sources)
    
    return {
        "answer": answer,
        "sources": sources,
        "cached": False
    }
