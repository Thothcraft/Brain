import requests
from typing import Dict, Any
from pydantic import BaseModel
from server.utils.functions_metadata import function_schema


class WeatherResponse(BaseModel):
    location: str
    forecast: str
    success: bool
    error: str = None


@function_schema(
    name="get_weather_forecast",
    description="Finds information the forecast of a specific location and provides a simple interpretation like, is going to rain, it's hot, it's super hot instead of warmer",
    required_params=["location"]
)
def get_weather_forecast(location: str) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location.
    
    Args:
        location: The location to get the weather forecast for
        
    Returns:
        Dict containing the weather forecast information
    """
    url = f"http://wttr.in/{location}?format=3"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Clean up the response text
        forecast = response.text.strip()
        
        return {
            "location": location,
            "forecast": forecast,
            "success": True
        }
        
    except requests.RequestException as e:
        return {
            "location": location,
            "forecast": "",
            "success": False,
            "error": f"Error getting weather data: {str(e)}"
        }