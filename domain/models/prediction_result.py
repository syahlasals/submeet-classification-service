# Domain Layer
# domain objects
from dataclasses import dataclass
from typing import Optional

@dataclass
class PredictionResult:
    paper_sub_id: int
    relevance_label: str  # 'in-scope' atau 'out_of_scope'
    predicted_topic: Optional[str]
    confidence_score: float
    model_label_raw: str