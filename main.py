# entry point FastAPI, tempat app dibuat dan router di-register
import os
import logging
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from api.routers import topic_detection_router
from infrastructure.ml.model_loader import ModelLoader

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cnn_path = os.getenv("CNN_MODEL_PATH", "infrastructure/model_files/bestmodel_glove_OA9.keras")
sigmas_path = os.getenv("SIGMAS_PATH", "infrastructure/model_files/sigmas_9.json")
vocab_path = os.getenv("VOCAB_PATH", "infrastructure/model_files/word_to_idx_9.json")
global_model_loader = ModelLoader(cnn_path, sigmas_path, vocab_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global_model_loader.load_all()
    app.state.model_loader = global_model_loader
    yield

app = FastAPI(title=os.getenv("APP_NAME", "SubMeet AI"), lifespan=lifespan)

# TEMPORARY MOCK WEBHOOK FOR TESTING (comment jika sudah tidak diperlukan)
@app.post("/api/mock-laravel-webhook")
async def mock_laravel_webhook(request: Request):
    payload = await request.json()
    logger.info("============== WEBHOOK DITERIMA DARI AI =============")
    logger.info(f"Payload JSON: {payload}")
    logger.info("=====================================================")
    return {"status": "success", "message": "Mock Laravel received the data!"}

# Daftarkan router
app.include_router(topic_detection_router.router, prefix="/api", tags=["Topic Detection"])