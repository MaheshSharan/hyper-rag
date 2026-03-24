# Pydantic models for Chunk, Document, etc.

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class ChunkMetadata(BaseModel):
    chunk_id: str
    source: str
    file_type: str
    language: Optional[str] = None
    page_number: Optional[int] = None
    section_path: List[str] = Field(default_factory=list)  # hierarchical path for PageIndex
    hierarchy_level: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: ChunkMetadata

class PageIndexNode(BaseModel):
    node_id: str
    title: str
    content: str
    level: int
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
