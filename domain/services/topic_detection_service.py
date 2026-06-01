import logging
import os
from config.settings import settings
from domain.models.prediction_result import PredictionResult
from infrastructure.preprocessing.text_preprocessor import TextPreprocessor
from infrastructure.ml.classification_model import ClassificationModel

logger = logging.getLogger(__name__)

class TopicDetectionService:
    def __init__(self, preprocessor: TextPreprocessor, classifier: ClassificationModel):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.model_name = self._generate_model_name()

    def _generate_model_name(self) -> str:
        try:
            model_path = settings.CNN_MODEL_PATH
            file_name = os.path.basename(model_path)
            model_name = os.path.splitext(file_name)[0]

            return model_name
        except Exception as e:
            logger.error(f"Gagal mengambil versi model: {e}")
            return "UNKNOWN_MODEL_VERSION"

    def process_text(self, paper_sub_id: int, title: str, abstract: str) -> PredictionResult:
        logger.info(f"Memulai analisis untuk Paper ID: {paper_sub_id}")
        
        try:
            # 1. Gabungkan Teks
            raw_text = f"{title} {abstract}"
            
            # 2. Preprocessing
            cleaned_text = self.preprocessor.process(raw_text)
            
            # 3. Klasifikasi
            result_dict = self.classifier.predict(cleaned_text)
            
            # 4. Petakan ke Domain Model
            return PredictionResult(
                paper_sub_id=paper_sub_id,
                relevance_label=result_dict["relevance_label"],
                predicted_topic=result_dict["predicted_topic"],
                confidence_score=result_dict["confidence_score"],
                model_label_raw=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Gagal memproses Paper ID {paper_sub_id}: {e}")
            raise