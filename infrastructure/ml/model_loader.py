import logging
import json
import os
import tensorflow as tf

logger = logging.getLogger(__name__)

class DynamicModelLoader:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.cache = {}

    def get_model(self, base_model: str):
        if base_model in self.cache:
            logger.info(f"Mengambil model '{base_model}' dari cache memori.")
            return self.cache[base_model]
        
        logger.info(f"Model '{base_model}' tidak ditemukan di cache. Memuat dari disk...")
        return self._load_model_from_disk(base_model)
    
    def _load_model_from_disk(self, base_model: str) -> dict:
        cnn_path = os.path.join(self.base_dir, f"{base_model}.keras")
        sigmas_path = os.path.join(self.base_dir, f"{base_model}_sigmas.json")
        vocab_path = os.path.join(self.base_dir, f"{base_model}_word_to_idx.json")
        
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"Model '{base_model}' tidak ditemukan.")
        if not os.path.exists(sigmas_path):
            raise FileNotFoundError(f"File sigmas '{base_model}_sigmas.json' tidak ditemukan.")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"File vocab '{base_model}_word_to_idx.json' tidak ditemukan.")
        
        try:
            cnn_model = tf.keras.models.load_model(cnn_path)
            with open(sigmas_path, 'r', encoding='utf-8') as f:
                sigmas = json.load(f)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)

            model_data = {
                "cnn_model": cnn_model,
                "sigmas": sigmas,
                "word_to_idx": vocab,
                "base_model": base_model
            }

            self.cache[base_model] = model_data
            logger.info(f"Model '{base_model}' berhasil dimuat dan disimpan ke cache.")
            return model_data
        
        except Exception as e:
            logger.error(f"Gagal memuat model '{base_model}': {e}")
            raise