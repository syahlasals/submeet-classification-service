# Technical Services
import logging
import json
import tensorflow as tf

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, cnn_path: str, sigmas_path: str):
        self.cnn_path = cnn_path
        self.sigmas_path = sigmas_path
        self.cnn_model = None
        self.sigmas = None

    def load_all(self):
        logger.info("Memuat CNN Model dan file Sigma dari memori...")
        # Load CNN Keras
        self.cnn_model = tf.keras.models.load_model(self.cnn_path)
        
        # Load nilai sigma untuk Gaussian Thresholding
        with open(self.sigmas_path, 'r') as f:
            self.sigmas = json.load(f)
            
        logger.info("CNN Model berhasil dimuat.")