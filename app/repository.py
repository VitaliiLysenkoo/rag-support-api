from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.models import DocumentChunk, SemanticCache
from typing import Optional

class DocumentRepository:
    @staticmethod
    async def delete_by_document_name(db: AsyncSession, document_name: str) -> None:
        """Deletes all chunks belonging to a specific document."""
        await db.execute(delete(DocumentChunk).where(DocumentChunk.document_name == document_name))
        await db.commit()

    @staticmethod
    async def get_all_document_names(db: AsyncSession) -> list[str]:
        """Returns a list of all unique document names currently in the database."""
        result = await db.execute(select(DocumentChunk.document_name).distinct())
        return [name[0] for name in result.all()]

    @staticmethod
    async def save_chunk(db: AsyncSession, document_name: str, page_number: int | None, content: str, embedding: list[float]) -> None:
        """Saves a single chunk to the database (without committing)."""
        db_chunk = DocumentChunk(
            document_name=document_name,
            page_number=page_number,
            content=content,
            embedding=embedding
        )
        db.add(db_chunk)

    @staticmethod
    async def get_similar_chunks(db: AsyncSession, query_vector: list[float], limit: int = 30):
        """Finds the closest vectors in the database using L2 distance (pgvector)."""
        result = await db.execute(
            select(DocumentChunk)
            .order_by(DocumentChunk.embedding.l2_distance(query_vector))
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def check_semantic_cache(db: AsyncSession, query_vector: list[float], max_distance: float = 0.05) -> Optional[SemanticCache]:
        """Checks if a semantically identical question was already answered."""
        result = await db.execute(
            select(SemanticCache)
            .where(SemanticCache.question_embedding.l2_distance(query_vector) < max_distance)
            .order_by(SemanticCache.question_embedding.l2_distance(query_vector))
            .limit(1)
        )
        return result.scalars().first()
        
    @staticmethod
    async def save_to_semantic_cache(db: AsyncSession, question: str, question_embedding: list[float], answer: str, sources: list[dict]) -> None:
        """Saves a generated answer to the semantic cache."""
        cache_entry = SemanticCache(
            question=question,
            question_embedding=question_embedding,
            answer=answer,
            sources=sources
        )
        db.add(cache_entry)
        await db.commit()

    @staticmethod
    async def clear_semantic_cache(db: AsyncSession) -> None:
        """Clears all entries from the semantic cache. Used when new documents are seeded."""
        await db.execute(delete(SemanticCache))
        await db.commit()
