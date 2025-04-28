"""External API integrations and tools for the Nomad Travel Agent."""

import datetime
import traceback
from typing import Any, Dict, Optional

import google.api_core.exceptions
import requests
from langchain_core.tools import tool
from retry import retry

from . import config

# --- API Error Handling ---
_RETRYABLE_ERRORS_GOOGLE = (
    google.api_core.exceptions.ResourceExhausted,
    google.api_core.exceptions.ServiceUnavailable,
    google.api_core.exceptions.InternalServerError,
    google.api_core.exceptions.DeadlineExceeded,
)


def get_google_retryable_exceptions():
    """Returns the tuple of Google API exceptions suitable for the retry decorator."""
    return _RETRYABLE_ERRORS_GOOGLE


def is_retriable_google_sdk(exception):
    """Predicate for retrying Google API calls."""
    return isinstance(exception, _RETRYABLE_ERRORS_GOOGLE)


# --- Unsplash Image Tool ---
@tool
@retry(
    tries=config.TOOL_RETRY_ATTEMPTS,
    delay=1,
    backoff=2,
    exceptions=requests.exceptions.RequestException,
)
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
        "content_filter": "high",
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
def get_placeholder_weather(
    city: str, country: Optional[str] = "Unknown"
) -> Dict[str, Any]:
    """Returns placeholder weather data dictionary."""
    print(f"Debug: Using placeholder weather for {city}")
    return {
        "location": f"{city}, {country} (Placeholder)",
        "temperature_celsius": None,
        "feels_like_celsius": None,
        "conditions": "Weather data unavailable",
        "humidity_percent": None,
        "wind_speed_mps": None,
        "timestamp": datetime.datetime.now().isoformat(),
        "is_placeholder": True,
    }


@tool
@retry(
    tries=config.TOOL_RETRY_ATTEMPTS,
    delay=1,
    backoff=2,
    exceptions=requests.exceptions.RequestException,
)
def get_weather(city: str, country: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches current weather from OpenWeatherMap and returns a structured dictionary.
    """
    api_key = config.OPENWEATHERMAP_API_KEY
    if not api_key:
        print("Debug: OpenWeatherMap key not found, using placeholder.")
        return get_placeholder_weather(city, country)

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    query = f"{city}{f',{country}' if country else ''}"
    params = {"q": query, "appid": api_key, "units": "metric"}

    print(f"Debug: Calling OpenWeatherMap for query: {query}")

    try:
        response = requests.get(base_url, params=params, timeout=10)
        print(f"Debug: OpenWeatherMap Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print(f"Debug: OpenWeatherMap Raw Data: {data}")

        main_data = data.get("main", {})
        weather_list = data.get("weather", [])
        wind_data = data.get("wind", {})
        sys_data = data.get("sys", {})

        condition_desc = "N/A"
        if weather_list and isinstance(weather_list[0], dict):
            condition_desc = weather_list[0].get("description", "N/A")

        weather_details = {
            "location": f"{data.get('name', city)}, {sys_data.get('country', country or 'Unknown')}",
            "temperature_celsius": main_data.get("temp"),
            "feels_like_celsius": main_data.get("feels_like"),
            "conditions": condition_desc.capitalize(),
            "humidity_percent": main_data.get("humidity"),
            "wind_speed_mps": wind_data.get("speed"),
            "pressure_hpa": main_data.get("pressure"),
            "timestamp": datetime.datetime.now().isoformat(),
            "is_placeholder": False,
        }
        print(f"Debug: Parsed weather details: {weather_details}")
        return weather_details

    except requests.exceptions.Timeout as e:
        print(f"Error fetching weather data (Timeout): {e}")
        return get_placeholder_weather(city, country)
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching weather data (HTTP Error {response.status_code}): {e}")
        if response.status_code == 404:
            placeholder = get_placeholder_weather(city, country)
            placeholder["conditions"] = "City not found by weather service"
            return placeholder
        return get_placeholder_weather(city, country)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data (Network Error): {e}")
        return get_placeholder_weather(city, country)
    except Exception as e:
        print(f"Unexpected error fetching weather data: {e}")
        traceback.print_exc()
        return get_placeholder_weather(city, country)


# --- Tool Lists ---
suggestion_tools = [unsplash_get_image]
phase2_direct_tools = {"get_weather": get_weather}
UNSPLASH_TOOL_NAME = unsplash_get_image.name
