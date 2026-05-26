import uuid
import os
from fastapi import APIRouter, BackgroundTasks, Depends, Request 
from schemas.request_schema import TopicDetectionRequest
from schemas.response_schema import StandardResponse
from domain.services.topic_detection_service import TopicDetectionService
from infrastructure.preprocessing.text_preprocessor import TextPreprocessor
from infrastructure.ml.classification_model import ClassificationModel

router = APIRouter()

def get_topic_service(request: Request):
    vocab_path = os.getenv("VOCAB_PATH", "infrastructure/model_files/word_to_idx.json")
    preprocessor = TextPreprocessor(vocab_path=vocab_path, max_seq_len=350, max_vocab=20000)
    loader = request.app.state.model_loader
    classifier = ClassificationModel(model_loader=loader, alpha=9.0)
    return TopicDetectionService(preprocessor, classifier)

@router.post("/detect", response_model=StandardResponse, status_code=202)
async def detect_topic(
    payload: TopicDetectionRequest, 
    background_tasks: BackgroundTasks,
    service: TopicDetectionService = Depends(get_topic_service)
):
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        service.process_and_notify,
        paper_sub_id=payload.paper_sub_id,
        title=payload.title,
        abstract=payload.abstract,
        webhook_url=payload.webhook_url
    )
    
    return StandardResponse(
        status="accepted",
        job_id=job_id,
        message="Permintaan klasifikasi sedang diproses di latar belakang."
    )