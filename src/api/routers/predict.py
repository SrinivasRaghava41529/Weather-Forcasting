#src/api/routers/predict.py

import logging
import uuid
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Security, Depends
from fastapi.security.api_key import APIKeyHeader

from src.config import cfg
from src.api.schemas.predict import PredictRequest, PredictResponse, HealthResponse

logger = logging.getLogger(__name__)

# Security Definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates the API Key before allowing prediction."""
    if api_key != cfg.api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Define the router
router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """Public endpoint to verify system status."""
    ml = request.app.state.ml_artifact
    is_loaded = "pipeline" in ml
    
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=ml.get("model_version", "unknown"),
        expected_features=len(ml.get("feature_cols", []))
    )

'''
@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)], tags=["Forecasting"])
async def predict_temperature(payload: PredictRequest, request: Request):
    """Protected endpoint: Generates forecasts from 34 features."""
    ml = request.app.state.ml_artifact
    
    if "pipeline" not in ml:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    
    expected_features = ml["feature_cols"]
    provided_features = set(payload.features.keys())
    
    # Validation: Ensure 100% feature match
    missing = set(expected_features) - provided_features
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {list(missing)}")
    
    try:
        df_input = pd.DataFrame([payload.features])[expected_features]
        
        # prediction is now an array: e.g., [16.1, 15.2, 12.0, 16.5, 14.8]
        prediction_array = ml["pipeline"].predict(df_input.values)[0]
        
        # Find which index corresponds to the user's requested horizon
        target_name = f"temp_c_next_{payload.horizon_hours}h"
        
        if target_name not in ml["target_cols"]:
             raise HTTPException(status_code=400, detail=f"Horizon {payload.horizon_hours} is not supported.")
             
        target_idx = ml["target_cols"].index(target_name)
        specific_prediction = prediction_array[target_idx]
        
        return PredictResponse(
            prediction_c=round(float(specific_prediction), 3),
            prediction_id=str(uuid.uuid4()),
            model_version=ml["model_version"]
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

'''

@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)], tags=["Forecasting"])
async def predict_temperature(payload: PredictRequest, request: Request):
    ml = request.app.state.ml_artifact
    
    if "pipeline" not in ml:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    
    # 1. Validate requested horizon
    target_name = f"temp_c_next_{payload.horizon_hours}h"
    if target_name not in ml["target_cols"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Horizon {payload.horizon_hours}h is not supported. Choose from: {ml['target_cols']}"
        )
        
    # 2. Validate features
    expected_features = ml["feature_cols"]
    provided_features = set(payload.features.keys())
    missing = set(expected_features) - provided_features
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {list(missing)}")
    
    # --- NO MORE TRY/EXCEPT. LET IT CRASH LOUDLY ---
    
    # 3. Inference
    df_input = pd.DataFrame([payload.features])[expected_features]
    prediction_array = ml["pipeline"].predict(df_input.values)[0]
    
    # 4. Extract the exact horizon requested
    target_idx = ml["target_cols"].index(target_name)
    specific_prediction = prediction_array[target_idx]
    
    return PredictResponse(
        horizon_hours=payload.horizon_hours,
        prediction_c=round(float(specific_prediction), 3),
        prediction_id=str(uuid.uuid4()),
        model_version=ml["model_version"]
    )