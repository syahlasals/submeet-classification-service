# entry point FastAPI, tempat app dibuat dan router di-register
import logging
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from uvicorn import config
from infrastructure.preprocessing.text_preprocessor import TextPreprocessor

from api.routers import topic_detection_router
from config.settings import settings
from infrastructure.ml.model_loader import ModelLoader

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global_model_loader = ModelLoader(
    cnn_path=settings.CNN_MODEL_PATH, 
    sigmas_path=settings.SIGMAS_PATH,
    # vocab_path=settings.VOCAB_PATH
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global_model_loader.load_all()
        app.state.model_loader = global_model_loader
        app.state.preprocessor = TextPreprocessor(
            vocab_path=settings.VOCAB_PATH,
            max_seq_len=settings.MAX_SEQ_LEN,
            max_vocab=settings.MAX_VOCAB
        )
        logger.info("Model dan preprocessor berhasil dimuat.")
    except Exception as e:
        logger.error(f"Gagal memuat model dan preprocessor: {e}")
        raise e
    yield

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# Daftarkan router
app.include_router(topic_detection_router.router, prefix="/api", tags=["Topic Detection"])