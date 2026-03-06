from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from backend.database import Base
from datetime import datetime, timezone

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    chunks = relationship("Chunk", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text_content = Column(Text)
    page_number = Column(Integer, nullable=True)
    
    # We use vector size 384 because the local sequence-transformer 'all-MiniLM-L6-v2' output is 384D.
    embedding = Column(Vector(384)) 
    
    document = relationship("Document", back_populates="chunks")
