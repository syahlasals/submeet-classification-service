import numpy as np
import logging

logger = logging.getLogger(__name__)

class ClassificationModel:
    def __init__(self, model_loader, alpha: float):
        self.cnn_model = model_loader.cnn_model
        self.sigmas = model_loader.sigmas
        self.alpha = alpha

    def predict(self, padded_sequence: np.ndarray) -> dict:
        probs = self.cnn_model.predict(padded_sequence, verbose=0)[0]
        
        max_class = int(np.argmax(probs))
        max_prob = float(probs[max_class])
        
        predicted_topic = f"Topic-{max_class}"
        sigma_i = 0.0
        
        try:
            if isinstance(self.sigmas, list):
                class_data = self.sigmas[max_class]
                if isinstance(class_data, dict):
                    # Ambil nama topik
                    if "label" in class_data:
                        predicted_topic = class_data["label"]
                    # Ambil sigma
                    if "sigma" in class_data:
                        sigma_i = float(class_data["sigma"])
        except Exception as e:
            logger.error(f"Gagal membaca data dari sigmas.json: {e}")
            
        # Hitung threshold
        threshold = max(0.5, 1.0 - (self.alpha * sigma_i))

        # debug log
        logger.info(f"Skor Probabilitas Topik Tertinggi ({predicted_topic}): {max_prob:.4f}")
        logger.info(f"Threshold yang harus dilewati: {threshold:.4f}")
        
        if max_prob >= threshold:
            logger.info("-> HASIL: IN-SCOPE (Diterima)")
            return {
                "relevance_label": "in_scope",
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