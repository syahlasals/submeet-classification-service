# Domain Layer
# domain objects
# from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, ConfigDict

class PredictionResult(BaseModel):
    paper_sub_id: int
    relevance_label: str  # 'in-scope' atau 'out_of_scope'
    predicted_topic: Optional[str]
    confidence_score: float
    model_label_raw: str

    model_config = ConfigDict(protected_namespaces=())