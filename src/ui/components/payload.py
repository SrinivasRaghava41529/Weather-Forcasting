def build_feature_payload(horizon_val, temp_val, humid_val, wind_val):
    """Converts simple UI inputs into the 34-feature baseline expected by the API."""
    return {
        "horizon_hours": horizon_val,
        "features": {
            "temp_c": temp_val,
            "humidity_pct": humid_val,
            "wind_speed": wind_val,
            "pressure_mbar": 1012.1,
            "dew_point_c": temp_val - 5.0, 
            "vapor_pressure_act": 12.0,
            "wind_speed_max": wind_val + 1.0,
            "wind_u": -1.2,
            "wind_v": 0.5,
            "pressure_tendency": 0.0,
            "hour_sin": 0.5, "hour_cos": 0.866,
            "month_sin": 0.866, "month_cos": 0.5,
            "dayofyear_sin": 0.1, "dayofyear_cos": 0.99,
            "is_daytime": 1,
            # Simulating a stable historical trend based on current temp
            "temp_c_lag_1h": temp_val - 0.2,
            "temp_c_lag_2h": temp_val - 0.3,
            "temp_c_lag_3h": temp_val - 0.5,
            "temp_c_lag_6h": temp_val - 1.0,
            "temp_c_lag_12h": temp_val - 3.0,
            "temp_c_lag_24h": temp_val + 0.1,
            "temp_c_lag_48h": temp_val + 0.2,
            "temp_c_roll_mean_3h": temp_val - 0.3,
            "temp_c_roll_std_3h": 0.1,
            "temp_c_roll_mean_6h": temp_val - 0.6,
            "temp_c_roll_std_6h": 0.2,
            "temp_c_roll_mean_12h": temp_val - 1.5,
            "temp_c_roll_std_12h": 1.0,
            "temp_c_roll_mean_24h": temp_val - 0.5,
            "temp_c_roll_std_24h": 2.0
        }
    }