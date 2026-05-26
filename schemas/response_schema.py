from pydantic import BaseModel

class StandardResponse(BaseModel):
    status: str
    job_id: str
    message: str