"""External API integrations and tools for the Nomad Travel Agent."""

import os
import requests
import datetime
import random
import traceback
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from retry import retry
import google.api_core.exceptions

from . import config

# --- API Error Handling ---
_RETRYABLE_ERRORS_GOOGLE = (
    google.api_core.exceptions.ResourceExhausted,
    google.api_core.exceptions.ServiceUnavailable,
    google.api_core.exceptions.InternalServerError,
    google.api_core.exceptions.DeadlineExceeded,
)

def is_retriable_google_sdk(exception):
    """Predicate for retrying Google API calls."""
    return isinstance(exception, _RETRYABLE_ERRORS_GOOGLE)

# --- Unsplash Image Tool ---
@tool
@retry(tries=config.TOOL_RETRY_ATTEMPTS, delay=1, backoff=2, exceptions=requests.exceptions.RequestException)
def unsplash_get_image(city: str, country: Optional[str] = None) -> str:
    """Fetches a relevant image URL from Unsplash for a given city."""
    access_key = config.UNSPLASH_ACCESS_KEY
    if not access_key:
        return config.PLACEHOLDER_IMAGE_URL

    search_query = f"{city}{f', {country}' if country else ''} city"
    url = "https://api.unsplash.com/photos/random"
    headers = {"Authorization": f"Client-ID {access_key}"}
    params = {
        "query": search_query,
        "orientation": "landscape",
        "content_filter": "high"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data["urls"]["regular"]
    except Exception as e:
        print(f"Error fetching Unsplash image: {e}")
        return config.PLACEHOLDER_IMAGE_URL

# --- OpenWeatherMap Tool ---
def get_placeholder_weather(city: str, country: Optional[str] = "Unknown") -> Dict[str, Any]:
    """Returns placeholder weather data when API is unavailable."""
    return {
        "location": f"{city}, {country}",
        "temperature": "N/A",
        "conditions": "Weather data unavailable",
        "humidity": "N/A",
        "wind_speed": "N/A",
        "timestamp": datetime.datetime.now().isoformat()
    }

@tool
@retry(tries=config.TOOL_RETRY_ATTEMPTS, delay=1, backoff=2, exceptions=requests.exceptions.RequestException)
def get_weather(city: str, country: Optional[str] = None) -> Dict[str, Any]:
    """Fetches current weather from OpenWeatherMap for a given city."""
    api_key = config.OPENWEATHERMAP_API_KEY
    if not api_key:
        return get_placeholder_weather(city, country)

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    query = f"{city}{f',{country}' if country else ''}"
    params = {
        "q": query,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        return {
            "location": f"{city}, {country if country else data.get('sys', {}).get('country', 'Unknown')}",
            "temperature": f"{data['main']['temp']}°C",
            "conditions": data['weather'][0]['description'],
            "humidity": f"{data['main']['humidity']}%",
            "wind_speed": f"{data['wind']['speed']} m/s",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return get_placeholder_weather(city, country)

# --- Tool Lists ---
suggestion_tools = [unsplash_get_image]
phase2_direct_tools = {'get_weather': get_weather}
UNSPLASH_TOOL_NAME = unsplash_get_image.name 