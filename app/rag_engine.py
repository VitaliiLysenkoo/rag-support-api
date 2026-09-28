import os
import fitz
from sqlalchemy.ext.asyncio import AsyncSession
from app.repository import DocumentRepository
from app.ai_service import get_embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import setup_logger
from app.database import AsyncSessionLocal

logger = setup_logger("rag_engine")

async def process_and_store_document(file_path: str):
    """
    Reads a file (TXT or PDF), chunks it semantically (page-by-page for PDFs), 
    gets embeddings from OpenAI, and stores them in PostgreSQL with page numbers.
    """
    async with AsyncSessionLocal() as db:
        document_name = os.path.basename(file_path)
        logger.info(f"Started processing document: {document_name}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        saved_chunks_count = 0

        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = text_splitter.split_text(text)
                for chunk_text in chunks:
                    if not chunk_text.strip(): continue
                    vector = await get_embedding(chunk_text)
                    await DocumentRepository.save_chunk(db, document_name, None, chunk_text, vector)
                    saved_chunks_count += 1

            elif file_path.endswith('.pdf'):
                with fitz.open(file_path) as pdf_doc:
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc[page_num]
                        page_text = page.get_text().replace('\n', ' ')
                        chunks = text_splitter.split_text(page_text)
                        for chunk_text in chunks:
                            if not chunk_text.strip(): continue
                            vector = await get_embedding(chunk_text)
                            await DocumentRepository.save_chunk(db, document_name, page_num + 1, chunk_text, vector)
                            saved_chunks_count += 1
            else:
                raise ValueError("Unsupported file format.")
                
            await db.commit()
            logger.info(f"Successfully processed and stored {saved_chunks_count} chunks for {document_name}")
            return saved_chunks_count
        except Exception as e:
            logger.error(f"Failed to process document {document_name}: {e}")
            await db.rollback()
            raise

async def search_similar_chunks(query: str, db: AsyncSession, limit: int = 3):
    """
    Converts user query into a vector and searches the database for similar chunks.
    """
    query_vector = await get_embedding(query)
    return await DocumentRepository.get_similar_chunks(db, query_vector, limit)
