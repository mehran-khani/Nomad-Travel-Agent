"""Data models and schemas for the Nomad Travel Agent."""

from typing import List, Dict, Any, Optional, Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

# --- Pydantic Models ---
# class ParsedPreference(BaseModel):
#     """User preferences parsed from conversation."""
#     vibe: str = Field(description="The desired atmosphere/vibe")
#     activities: str = Field(description="Preferred activities")
#     weather: str = Field(description="Weather preferences")
#     budget: str = Field(description="Budget level")
#     has_all_preferences: bool = Field(description="Whether all preferences are collected")


class ParsedPreference(BaseModel):
    """Structure for parsing a single preference identified in the user's last message."""

    preference_key: Optional[str] = Field(
        default=None,
        description="The category of preference identified (e.g., 'vibe', 'activities', 'weather', 'budget'), or null if none found.",
    )
    preference_value: Optional[str] = Field(
        default=None,
        description="The specific value of the preference stated by the user, or null if none found.",
    )


class RecommendationPydantic(BaseModel):
    """Single city recommendation."""

    city: str
    country: str
    description: str
    justification: str
    has_data: bool = True


class CityRecommendationsList(BaseModel):
    """List of city recommendations."""

    recommendations: List[RecommendationPydantic]


class PointOfInterest(BaseModel):
    """Details about a point of interest."""

    name: str
    type: str
    description: str
    address: Optional[str] = None
    rating: Optional[float] = None


class CityEvent(BaseModel):
    """Details about a city event."""

    name: str
    description: str
    date: Optional[str] = None
    location: Optional[str] = None


class CityInformation(BaseModel):
    """Comprehensive city information."""

    city_name: str
    country: str
    overview: str
    weather_summary: Optional[str] = None
    points_of_interest: List[PointOfInterest]
    events: List[CityEvent]
    local_tips: List[str]
    best_times_to_visit: List[str]


# --- LangGraph State Types ---
class Recommendation(TypedDict):
    """Phase 1 final recommendation output."""

    city: str
    country: str
    description: str
    justification: str
    image_url: Optional[str]
    has_data: bool


class SuggestionState(TypedDict):
    """State for the suggestion phase of conversation."""

    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    user_preferences: Dict[str, Any]
    text_recommendations: Optional[List[Dict]]
    recommendations: List[Recommendation]
    cities_with_data: List[str]
    error_message: Optional[str]
    is_finished: bool
    selected_city_for_phase_2: Optional[str]
