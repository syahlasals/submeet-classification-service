import numpy as np
import logging
from domain.models.topic_category import ISSAT_TOPICS

logger = logging.getLogger(__name__)

class ClassificationModel:
    def __init__(self, model_loader, alpha: float = 9.0):
        self.cnn_model = model_loader.cnn_model
        self.sigmas = model_loader.sigmas
        self.alpha = alpha

    def predict(self, padded_sequence: np.ndarray) -> dict:
        probs = self.cnn_model.predict(padded_sequence, verbose=0)[0]
        
        max_class = int(np.argmax(probs))
        max_prob = float(probs[max_class])
        
        # debug log
        logger.info(f"Skor Probabilitas Topik Tertinggi ({ISSAT_TOPICS[max_class]}): {max_prob:.4f}")
        
        # default threshold minimum
        threshold = 0.5
        
        try:
            if isinstance(self.sigmas, list):
                class_data = self.sigmas[max_class]
                if isinstance(class_data, dict) and "threshold" in class_data:
                    threshold = float(class_data["threshold"])
        except Exception as e:
            logger.error(f"Gagal membaca threshold: {e}")
            
        logger.info(f"Threshold yang harus dilewati: {threshold:.4f}")
        
        if max_prob >= threshold:
            logger.info("-> HASIL: IN-SCOPE (Diterima)")
            predicted_topic = ISSAT_TOPICS[max_class]
            return {
                "relevance_label": "relevant",
                "predicted_topic": predicted_topic,
                "confidence_score": max_prob
            }
        else:
            logger.info("-> HASIL: OUT-OF-SCOPE (Ditolak)")
            return {
                "relevance_label": "out_of_scope",
                "predicted_topic": None,
                "confidence_score": max_prob
            }