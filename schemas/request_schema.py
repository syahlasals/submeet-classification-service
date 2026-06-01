from pydantic import BaseModel

class TopicDetectionRequest(BaseModel):
    paper_sub_id: int
    title: str
    abstract: str