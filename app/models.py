from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

# --- Request Models ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the document.")
    collection_name: str = Field(..., min_length=1, description="The name of the collection (derived from filename usually) to query.")
    n_results: int = Field(default=3, gt=0, le=10, description="Number of relevant chunks to retrieve.")
    # Example of a simple 'where' filter structure
    where_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter for ChromaDB query.")

    @validator('collection_name')
    def sanitize_collection_name(cls, v):
        # Basic sanitization, adjust as needed
        return v.strip().lower().replace(" ", "_")

# --- Response Models ---

class UploadResponse(BaseModel):
    success: bool = True
    message: str
    collection_name: str
    chunks_added: int

class AskResponse(BaseModel):
    answer: str
    question: str
    collection_name: str
    context_used: List[str] = Field(description="List of document chunks used as context for the answer.")

class ErrorResponse(BaseModel):
    error: str