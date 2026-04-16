import streamlit as st

def render_sidebar():
    """Renders the sidebar controls and returns the current environmental inputs."""
    with st.sidebar:
        st.header("🌡️ Current Conditions")
        
        current_temp = st.slider("Temperature (°C)", min_value=-10.0, max_value=40.0, value=15.5, step=0.1)
        current_humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        current_wind = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=1.3, step=0.1)
        
        return current_temp, current_humidity, current_wind