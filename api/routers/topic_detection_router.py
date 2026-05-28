from fastapi import APIRouter, Depends, Request 
from schemas.request_schema import TopicDetectionRequest
from domain.models.prediction_result import PredictionResult 
from domain.services.topic_detection_service import TopicDetectionService
from infrastructure.preprocessing.text_preprocessor import TextPreprocessor
from infrastructure.ml.classification_model import ClassificationModel

router = APIRouter()

def get_topic_service(request: Request) -> TopicDetectionService:
    loader = request.app.state.model_loader
    preprocessor = request.app.state.preprocessor
    classifier = ClassificationModel(model_loader=loader, alpha=9.0)
    return TopicDetectionService(preprocessor, classifier)

@router.post("/detect", response_model=PredictionResult, status_code=200)
async def detect_topic(
    payload: TopicDetectionRequest, 
    service: TopicDetectionService = Depends(get_topic_service)
):
    result = service.process_text(
        paper_sub_id=payload.paper_sub_id,
        title=payload.title,
        abstract=payload.abstract
    )
    
    return result