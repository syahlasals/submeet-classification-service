# entry point FastAPI, tempat app dibuat dan router di-register
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from api.routers import topic_detection_router
from config.settings import settings
from infrastructure.ml.model_loader import DynamicModelLoader

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global_model_loader = DynamicModelLoader(base_dir=settings.MODEL_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global_model_loader.get_model(settings.DEFAULT_BASE_MODEL)
        app.state.model_loader = global_model_loader
        logger.info("Model berhasil diinisialisasi.")
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        raise e
    yield

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# Daftarkan router
app.include_router(topic_detection_router.router, prefix="/api", tags=["Topic Detection"])