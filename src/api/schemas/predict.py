#src/api/schemas/predict.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict
from datetime import datetime

class PredictRequest(BaseModel):
    # Field 1: The horizon choice
    horizon_hours: int = Field(
        ..., 
        description="Forecast horizon. Must be one of: 1, 6, 12, 24, 48"
    )
    
    # Field 2: The features dictionary (This is what was missing!)
    features: Dict[str, float] = Field(
        ..., 
        description="Key-value pairs of the required features."
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "horizon_hours": 24,
                "features": {
                    "temp_c": 15.5,
                    "pressure_mbar": 1012.1,
                    "humidity_pct": 70.0,
                    "dew_point_c": 10.0,
                    "vapor_pressure_act": 12.0,
                    "wind_speed": 1.3,
                    "wind_speed_max": 2.0,
                    "wind_u": -1.2,
                    "wind_v": 0.5,
                    "pressure_tendency": 0.0,
                    "hour_sin": 0.5,
                    "hour_cos": 0.866,
                    "month_sin": 0.866,
                    "month_cos": 0.5,
                    "dayofyear_sin": 0.1,
                    "dayofyear_cos": 0.99,
                    "is_daytime": 1,
                    "temp_c_lag_1h": 15.2,
                    "temp_c_lag_2h": 15.1,
                    "temp_c_lag_3h": 15.0,
                    "temp_c_lag_6h": 14.5,
                    "temp_c_lag_12h": 12.0,
                    "temp_c_lag_24h": 15.4,
                    "temp_c_lag_48h": 15.6,
                    "temp_c_roll_mean_3h": 15.2,
                    "temp_c_roll_std_3h": 0.1,
                    "temp_c_roll_mean_6h": 15.0,
                    "temp_c_roll_std_6h": 0.2,
                    "temp_c_roll_mean_12h": 13.5,
                    "temp_c_roll_std_12h": 1.5,
                    "temp_c_roll_mean_24h": 14.0,
                    "temp_c_roll_std_24h": 2.0
                }
            }
        }
    )

class PredictResponse(BaseModel):
    horizon_hours: int = Field(..., description="The requested forecast horizon")
    prediction_c: float = Field(..., description="Forecasted temperature in Celsius")
    prediction_id: str = Field(..., description="Trace ID for auditing")
    model_version: str
    served_at: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    expected_features: int
    supported_horizons: list[str]