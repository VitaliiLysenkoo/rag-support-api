from sqlalchemy import Column, Integer, String, Text, JSON
from pgvector.sqlalchemy import Vector
from app.database import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_name = Column(String, index=True, nullable=False)
    page_number = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    # 1536 is the vector dimension for OpenAI's text-embedding-3-small model
    embedding = Column(Vector(1536))

class SemanticCache(Base):
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    # Vector signature for the semantic question
    question_embedding = Column(Vector(1536), nullable=False)
    answer = Column(Text, nullable=False)
    # Storing structured JSON of the used sources
    sources = Column(JSON, nullable=False)
