from pydantic import BaseModel
from typing import Optional

class TopicDetectionRequest(BaseModel):
    paper_sub_id: int
    title: str
    abstract: str
    base_model: Optional[str] = None