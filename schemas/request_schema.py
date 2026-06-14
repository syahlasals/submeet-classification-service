from pydantic import BaseModel, Field, field_validator
from typing import Optional

class TopicDetectionRequest(BaseModel):
    paper_sub_id: int = Field(..., gt=0)
    title: str = Field(..., strip_whitespace=True)
    abstract: str = Field(..., strip_whitespace=True)
    base_model: str = Field(..., strip_whitespace=True)

    @field_validator('title', 'abstract', 'base_model')
    @classmethod
    def prevent_empty_strings(cls, value: str, info) -> str:
        field_name = info.field_name.replace('_', ' ').title()
        
        if not value or not value.strip():
            raise ValueError(f"{field_name} wajib diisi dan tidak boleh kosong.")
        
        return value

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, value):
      return value.strip()