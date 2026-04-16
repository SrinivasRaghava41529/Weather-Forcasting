#src/api/main.py

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config import cfg
from src.models.train import load_model
from src.api.routers import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model into app.state exactly once at startup."""
    logger.info("Starting up API and loading ML artifact...")
    app.state.ml_artifact = {}
    
    try:
        loaded = load_model()
        app.state.ml_artifact["pipeline"] = loaded["pipeline"]
        app.state.ml_artifact["feature_cols"] = loaded["feature_cols"]
        app.state.ml_artifact["model_version"] = loaded["model_version"]
        
        # FIXED: Tell the API to load the target_cols into memory!
        app.state.ml_artifact["target_cols"] = loaded["target_cols"]
        
        logger.info(f"Artifact loaded. Expecting {len(loaded['feature_cols'])} features.")
        logger.info(f"Supported horizons: {loaded['target_cols']}")
    except Exception as e:
        logger.error(f"Failed to load model artifact: {e}")
    
    yield
    app.state.ml_artifact.clear()

# Initialize App
app = FastAPI(
    title="Intelligent Forecasting API",
    version=cfg.model_version,
    lifespan=lifespan
)

# Apply CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire up the Router
app.include_router(predict.router)