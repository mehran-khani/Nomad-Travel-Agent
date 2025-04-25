"""HTML display functions for the Nomad Travel Agent."""

from typing import List, Dict, Any, Optional
from html import escape
from langchain_core.messages import ToolMessage, AIMessage

from . import config


def display_recommendations_html(
    text_recommendations: List[Dict], messages: List[Any]
) -> str:
    """Generates HTML for displaying Phase 1 recommendations."""
    # Extract tool results from messages
    tool_results = {}
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = msg.content
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            break

    # Build HTML
    html_parts = [
        '<div style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto;">'
    ]
    html_parts.append('<h2 style="color: #2c3e50;">🌍 Travel Recommendations</h2>')

    for i, rec in enumerate(text_recommendations, 1):
        city = escape(rec.get("city", "Unknown City"))
        country = escape(rec.get("country", ""))
        description = escape(rec.get("description", ""))
        justification = escape(rec.get("justification", ""))

        # Get image URL from tool results
        expected_call_id = f"{config.UNSPLASH_TOOL_NAME}_{i}_{city.replace(' ','_')}"
        image_url = tool_results.get(expected_call_id, config.PLACEHOLDER_IMAGE_URL)

        html_parts.append(
            f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; margin: 15px 0; padding: 15px; background: white;">
                <img src="{image_url}" alt="{city}" style="width: 100%; height: 300px; object-fit: cover; border-radius: 4px;">
                <h3 style="color: #34495e; margin: 10px 0;">{city}, {country}</h3>
                <p style="color: #7f8c8d; margin: 10px 0;"><strong>Overview:</strong> {description}</p>
                <p style="color: #95a5a6;"><strong>Why Visit:</strong> {justification}</p>
            </div>
        """
        )

    html_parts.append("</div>")
    return "\n".join(html_parts)


def display_city_details_html(
    city_info_dict: Dict[str, Any], selected_city_img_url: Optional[str]
) -> str:
    """
    Generates a complete HTML5 document string for displaying Phase 2 city details,
    including a detailed weather breakdown from the 'weather_details' dictionary.
    """

    # --- 1. Safe Data Extraction and Escaping ---
    city_name = escape(
        city_info_dict.get("city_name", "Unknown City") or "Unknown City"
    )
    country = escape(city_info_dict.get("country_name", "") or "")
    img_url = escape(selected_city_img_url or config.PLACEHOLDER_IMAGE_URL)
    overview = escape(city_info_dict.get("general_summary", "") or "")

    weather_details = city_info_dict.get("weather_details", {})

    pois_raw = city_info_dict.get("points_of_interest", [])
    pois = pois_raw if isinstance(pois_raw, list) else []

    events_raw = city_info_dict.get("events", [])
    events = events_raw if isinstance(events_raw, list) else []

    # --- 2. Start HTML Document Construction ---
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"<title>Explore {city_name}</title>",
        """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"; line-height: 1.6; background-color: #f4f7f6; margin: 0; padding: 20px; color: #333; }
            .container { max-width: 900px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 3px 15px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden; }
            .header-image-container { position: relative; }
            .header-image { width: 100%; height: 350px; object-fit: cover; display: block; border-bottom: 1px solid #eee; }
            .header-overlay { position: absolute; bottom: 0; left: 0; right: 0; background: linear-gradient(to top, rgba(0,0,0,0.75) 0%, rgba(0,0,0,0) 100%); padding: 40px 30px 20px 30px; }
            .header-overlay h1 { color: white; margin: 0; font-size: 2.8em; text-shadow: 1px 1px 3px rgba(0,0,0,0.5); }
            .content-section { padding: 25px 30px; border-bottom: 1px solid #eee; }
            .content-section:last-child { border-bottom: none; }
            .content-section h2 { color: #2c3e50; margin-top: 0; margin-bottom: 15px; font-size: 1.8em; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; display: inline-block;}
            .content-section p, .content-section li { margin: 0.6em 0; }
            /* Changed weather section class */
            .weather-details-section { background-color: #eef7fa; border-left: 4px solid #17a2b8; padding: 15px 20px; margin: 20px 30px; border-radius: 4px;}
            .weather-details-section h2 { border: none; margin-bottom: 10px; font-size: 1.6em; color: #106a7a;}
            .weather-details-section ul { list-style: none; padding: 0; margin: 0; }
            .weather-details-section li { margin-bottom: 8px; font-size: 1em; border-bottom: 1px solid #d1eaf0; padding-bottom: 8px;}
            .weather-details-section li:last-child { border-bottom: none; padding-bottom: 0; margin-bottom: 0;}
            .weather-details-section strong { color: #106a7a; min-width: 110px; display: inline-block;} /* Weather label styling */
            .poi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
            .poi-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 15px 20px; background: #fdfdfd; transition: box-shadow 0.2s ease-in-out; }
            .poi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            .poi-card h3 { color: #0056b3; margin: 0 0 8px 0; font-size: 1.25em; }
            .poi-card p { margin: 4px 0; font-size: 0.95em; }
            .poi-card strong { color: #555; font-weight: 600; }
            .event-list { list-style: none; padding: 0; margin: 0; }
            .event-item { margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px dashed #ccc; }
            .event-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0;}
            .event-item h3 { color: #5a3b8a; margin: 0 0 5px 0; font-size: 1.25em;}
            .event-item p { margin: 3px 0; font-size: 0.95em; }
            .placeholder-notice { font-size: 0.9em; color: #6c757d; font-style: italic; margin-top: 10px; }
        </style>
        """,
        "</head>",
        "<body>",
        '    <div class="container">',
    ]

    # --- 3. Add Header Section ---
    html_parts.append(
        f"""
        <div class="header-image-container">
            <img src="{img_url}" alt="Image of {city_name}" class="header-image">
            <div class="header-overlay">
                <h1>{city_name}{f", {country}" if country else ""}</h1>
            </div>
        </div>
    """
    )

    # --- 4. Add Overview Section ---
    if overview:
        html_parts.append(
            f"""
        <div class="content-section">
            <h2>📝 Overview</h2>
            <p>{overview}</p>
        </div>
        """
        )

    # --- 5. Add Detailed Weather Section ---
    html_parts.append(
        '<div class="content-section weather-details-section">'
    )  # Changed class name
    html_parts.append("<h2>🌤️ Weather Details</h2>")
    if weather_details and isinstance(weather_details, dict):
        html_parts.append("<ul>")

        # Define order and formatting for weather details
        weather_display_order = [
            ("location", "Location"),
            ("conditions", "Conditions"),
            ("temperature_celsius", "Temperature"),
            ("feels_like_celsius", "Feels Like"),
            ("humidity_percent", "Humidity"),
            ("wind_speed_mps", "Wind Speed"),
            ("pressure_hpa", "Pressure"),
        ]

        for key, label in weather_display_order:
            value = weather_details.get(key)
            display_value = ""
            unit = ""

            if value is not None:
                if key in ["temperature_celsius", "feels_like_celsius"]:
                    unit = "°C"
                    try:
                        display_value = f"{float(value):.1f}"
                    except (ValueError, TypeError):
                        display_value = escape(str(value))
                elif key == "humidity_percent":
                    unit = "%"
                    try:
                        display_value = f"{int(value)}"
                    except (ValueError, TypeError):
                        display_value = escape(str(value))
                elif key == "wind_speed_mps":
                    unit = " m/s"
                    try:
                        display_value = f"{float(value):.1f}"
                    except (ValueError, TypeError):
                        display_value = escape(str(value))
                elif key == "pressure_hpa":
                    unit = " hPa"
                    try:
                        display_value = f"{int(value)}"
                    except (ValueError, TypeError):
                        display_value = escape(str(value))
                else:
                    display_value = escape(str(value))

                html_parts.append(
                    f"<li><strong>{label}:</strong> {display_value}{unit}</li>"
                )

        html_parts.append("</ul>")

        if weather_details.get("is_placeholder"):
            html_parts.append(
                '<p class="placeholder-notice">(Weather data is placeholder)</p>'
            )

    else:
        html_parts.append("<p>Weather information currently unavailable.</p>")

    html_parts.append("</div>")

    # --- 6. Add Points of Interest Section ---
    if pois:
        html_parts.append(
            """
        <div class="content-section">
            <h2>🎯 Points of Interest</h2>
            <div class="poi-grid">
        """
        )
        for poi in pois:
            if not isinstance(poi, dict):
                continue
            name = escape(poi.get("name", "") or "")
            poi_type = escape(poi.get("type", "") or "")
            description = escape(poi.get("description", "") or "")
            address_raw = poi.get("address")
            address = escape(address_raw or "")
            if not name:
                continue
            type_html = f"<p><strong>Type:</strong> {poi_type}</p>" if poi_type else ""
            description_html = f"<p>{description}</p>" if description else ""
            address_html = (
                f'<p><strong style="color: #6c757d;">Address:</strong> {address}</p>'
                if address_raw
                else ""
            )
            html_parts.append(
                f"""
                <div class="poi-card">
                    <h3>{name}</h3>
                    {type_html}
                    {description_html}
                    {address_html}
                </div>
            """
            )
        html_parts.append("</div></div>")
    else:
        html_parts.append(
            """
        <div class="content-section">
            <h2>🎯 Points of Interest</h2>
            <p>No specific points of interest identified based on your preferences.</p>
        </div>
        """
        )

    # --- 7. Add Events Section ---
    if events:
        html_parts.append(
            """
        <div class="content-section">
            <h2>📅 Events</h2>
            <ul class="event-list">
        """
        )
        for event in events:
            if not isinstance(event, dict):
                continue
            name = escape(event.get("name", "") or "")
            summary = escape(event.get("summary", "") or "")
            if not name:
                continue
            summary_html = f"<p>{summary}</p>" if summary else ""
            html_parts.append(
                f"""
                <li class="event-item">
                    <h3>{name}</h3>
                    {summary_html}
                </li>
            """
            )
        html_parts.append("</ul></div>")
    else:
        html_parts.append(
            """
        <div class="content-section">
            <h2>📅 Events</h2>
            <p>No specific relevant events were found at this time.</p>
        </div>
        """
        )

    # --- 8. Add Closing Tags ---
    html_parts.append("    </div>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    # --- 9. Return the Complete HTML String ---
    return "\n".join(html_parts)


def display_qna_answer_html(llm_answer_text: str) -> str:
    """Generates HTML for displaying a Q&A answer."""
    formatted_answer = escape(llm_answer_text).replace("\n", "<br>")
    return f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto;">
            <div style="background: white; padding: 20px; border-radius: 8px; 
                        border-left: 4px solid #3498db; margin-bottom: 20px;">
                <p style="color: #34495e; margin: 0;">{formatted_answer}</p>
            </div>
        </div>
    """
