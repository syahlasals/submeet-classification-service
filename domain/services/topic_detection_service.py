import httpx
import logging
import uuid
from domain.models.prediction_result import PredictionResult
from infrastructure.preprocessing.text_preprocessor import TextPreprocessor
from infrastructure.ml.classification_model import ClassificationModel

logger = logging.getLogger(__name__)

class TopicDetectionService:
    def __init__(self, preprocessor: TextPreprocessor, classifier: ClassificationModel):
        self.preprocessor = preprocessor
        self.classifier = classifier

    async def process_and_notify(self, paper_sub_id: int, title: str, abstract: str, webhook_url: str):
        logger.info(f"Memulai analisis untuk Paper ID: {paper_sub_id}")
        
        try:
            # 1. Gabungkan Teks
            raw_text = f"{title}. {abstract}"
            
            # 2. Panggil Technical Service: Preprocessing
            cleaned_text = self.preprocessor.process(raw_text)
            
            # 3. Panggil Technical Service: ML Prediction (CNN + DOC)
            result_dict = self.classifier.predict(cleaned_text)
            
            # 4. Petakan ke Domain Model
            prediction = PredictionResult(
                paper_sub_id=paper_sub_id,
                relevance_label=result_dict["relevance_label"],
                predicted_topic=result_dict["predicted_topic"],
                confidence_score=result_dict["confidence_score"],
                model_label_raw="CNN_DOC_ALPHA_5.0"
            )
            
            # 5. Kirim Webhook (Technical Service via HTTPX)
            await self._send_webhook(webhook_url, prediction)
            
        except Exception as e:
            logger.error(f"Gagal memproses Paper ID {paper_sub_id}: {str(e)}")

    async def _send_webhook(self, url: str, prediction: PredictionResult):
        payload = {
            "paper_sub_id": prediction.paper_sub_id,
            "relevance_label": prediction.relevance_label,
            "predicted_topic": prediction.predicted_topic,
            "confidence_score": prediction.confidence_score,
            "model_label_raw": prediction.model_label_raw
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=30.0)
            logger.info(f"Webhook terkirim. Status HTTP: {response.status_code}")