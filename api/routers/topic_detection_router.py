from fastapi import APIRouter, Depends, HTTPException, Request 
from config.settings import settings
from schemas.request_schema import TopicDetectionRequest
from domain.models.prediction_result import PredictionResult 
from domain.services.topic_detection_service import TopicDetectionService
from infrastructure.preprocessing.text_preprocessor import TextPreprocessor
from infrastructure.ml.classification_model import ClassificationModel

class TopicDetectionRouter:
    router = APIRouter()

    @staticmethod
    def get_model_loader(request: Request):
        return request.app.state.model_loader

    @staticmethod
    def get_topic_service(target_model_name: str, loader) -> TopicDetectionService:
        try:
            model_data = loader.get_model(target_model_name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        preprocessor = TextPreprocessor(
            vocab=model_data["word_to_idx"],
            max_seq_len=settings.MAX_SEQ_LEN,
            max_vocab=settings.MAX_VOCAB
        )

        classifier = ClassificationModel(
            cnn_model=model_data["cnn_model"], 
            sigmas=model_data["sigmas"], 
            alpha=settings.ALPHA,
            base_model=model_data["base_model"]
        )
        
        return TopicDetectionService(preprocessor, classifier)

    @router.post("/detect", response_model=PredictionResult, status_code=200)
    @staticmethod
    def detect_topic(
        payload: TopicDetectionRequest, 
        loader = Depends(get_model_loader)
    ):
        service = TopicDetectionRouter.get_topic_service(payload.base_model, loader)
        
        result = service.process_text(
            paper_sub_id=payload.paper_sub_id,
            title=payload.title,
            abstract=payload.abstract
        )
        
        return result

router = TopicDetectionRouter.router