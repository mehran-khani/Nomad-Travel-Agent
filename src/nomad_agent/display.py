"""HTML display functions for the Nomad Travel Agent."""

from typing import List, Dict, Any, Optional
from html import escape
from langchain_core.messages import ToolMessage, AIMessage

from . import config

def display_recommendations_html(text_recommendations: List[Dict], messages: List[Any]) -> str:
    """Generates HTML for displaying Phase 1 recommendations."""
    # Extract tool results from messages
    tool_results = {}
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = msg.content
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            break

    # Build HTML
    html_parts = ['<div style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto;">']
    html_parts.append('<h2 style="color: #2c3e50;">🌍 Travel Recommendations</h2>')

    for i, rec in enumerate(text_recommendations, 1):
        city = escape(rec.get('city', 'Unknown City'))
        country = escape(rec.get('country', ''))
        description = escape(rec.get('description', ''))
        justification = escape(rec.get('justification', ''))
        
        # Get image URL from tool results
        expected_call_id = f"{config.UNSPLASH_TOOL_NAME}_{i}_{city.replace(' ','_')}"
        image_url = tool_results.get(expected_call_id, config.PLACEHOLDER_IMAGE_URL)
        
        html_parts.append(f'''
            <div style="border: 1px solid #ddd; border-radius: 8px; margin: 15px 0; padding: 15px; background: white;">
                <img src="{image_url}" alt="{city}" style="width: 100%; height: 300px; object-fit: cover; border-radius: 4px;">
                <h3 style="color: #34495e; margin: 10px 0;">{city}, {country}</h3>
                <p style="color: #7f8c8d; margin: 10px 0;"><strong>Overview:</strong> {description}</p>
                <p style="color: #95a5a6;"><strong>Why Visit:</strong> {justification}</p>
            </div>
        ''')

    html_parts.append('</div>')
    return '\n'.join(html_parts)

def display_city_details_html(city_info_dict: Dict[str, Any], selected_city_img_url: Optional[str]) -> str:
    """Generates HTML for displaying Phase 2 city details."""
    html_parts = ['<div style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto;">']
    
    # Header with image
    city_name = escape(city_info_dict.get('city_name', 'Unknown City'))
    country = escape(city_info_dict.get('country', ''))
    html_parts.append(f'''
        <div style="position: relative; margin-bottom: 20px;">
            <img src="{selected_city_img_url}" alt="{city_name}" 
                 style="width: 100%; height: 400px; object-fit: cover; border-radius: 8px;">
            <div style="position: absolute; bottom: 0; left: 0; right: 0; 
                        background: linear-gradient(transparent, rgba(0,0,0,0.7));
                        padding: 20px; border-radius: 0 0 8px 8px;">
                <h1 style="color: white; margin: 0;">{city_name}, {country}</h1>
            </div>
        </div>
    ''')

    # Overview
    overview = escape(city_info_dict.get('overview', ''))
    html_parts.append(f'''
        <div style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="color: #2c3e50;">Overview</h2>
            <p style="color: #34495e;">{overview}</p>
        </div>
    ''')

    # Weather
    weather = escape(city_info_dict.get('weather_summary', 'Weather information unavailable'))
    html_parts.append(f'''
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="color: #2c3e50;">🌤️ Weather</h2>
            <p style="color: #34495e;">{weather}</p>
        </div>
    ''')

    # Points of Interest
    pois = city_info_dict.get('points_of_interest', [])
    if pois:
        html_parts.append('''
            <div style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="color: #2c3e50;">🎯 Points of Interest</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px;">
        ''')
        for poi in pois:
            name = escape(poi.get('name', ''))
            type_ = escape(poi.get('type', ''))
            desc = escape(poi.get('description', ''))
            addr = escape(poi.get('address', ''))
            html_parts.append(f'''
                <div style="border: 1px solid #ddd; padding: 15px; border-radius: 4px;">
                    <h3 style="color: #34495e; margin: 0 0 10px 0;">{name}</h3>
                    <p style="color: #7f8c8d; margin: 5px 0;"><strong>Type:</strong> {type_}</p>
                    <p style="color: #34495e; margin: 5px 0;">{desc}</p>
                    {f'<p style="color: #95a5a6; margin: 5px 0;"><strong>Address:</strong> {addr}</p>' if addr else ''}
                </div>
            ''')
        html_parts.append('</div></div>')

    # Events
    events = city_info_dict.get('events', [])
    if events:
        html_parts.append('''
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="color: #2c3e50;">📅 Events</h2>
        ''')
        for event in events:
            name = escape(event.get('name', ''))
            desc = escape(event.get('description', ''))
            date = escape(event.get('date', ''))
            location = escape(event.get('location', ''))
            html_parts.append(f'''
                <div style="margin-bottom: 15px;">
                    <h3 style="color: #34495e; margin: 0 0 5px 0;">{name}</h3>
                    {f'<p style="color: #7f8c8d; margin: 5px 0;"><strong>Date:</strong> {date}</p>' if date else ''}
                    {f'<p style="color: #7f8c8d; margin: 5px 0;"><strong>Location:</strong> {location}</p>' if location else ''}
                    <p style="color: #34495e; margin: 5px 0;">{desc}</p>
                </div>
            ''')
        html_parts.append('</div>')

    # Local Tips and Best Times
    tips = city_info_dict.get('local_tips', [])
    best_times = city_info_dict.get('best_times_to_visit', [])
    if tips or best_times:
        html_parts.append('''
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
        ''')
        if tips:
            html_parts.append('''
                <div style="background: white; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #2c3e50;">💡 Local Tips</h2>
                    <ul style="color: #34495e; padding-left: 20px;">
            ''')
            for tip in tips:
                html_parts.append(f'<li>{escape(tip)}</li>')
            html_parts.append('</ul></div>')
        
        if best_times:
            html_parts.append('''
                <div style="background: white; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #2c3e50;">🗓️ Best Times to Visit</h2>
                    <ul style="color: #34495e; padding-left: 20px;">
            ''')
            for time in best_times:
                html_parts.append(f'<li>{escape(time)}</li>')
            html_parts.append('</ul></div>')
        html_parts.append('</div>')

    html_parts.append('</div>')
    return '\n'.join(html_parts)

def display_qna_answer_html(llm_answer_text: str) -> str:
    """Generates HTML for displaying a Q&A answer."""
    formatted_answer = escape(llm_answer_text).replace('\n', '<br>')
    return f'''
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto;">
            <div style="background: white; padding: 20px; border-radius: 8px; 
                        border-left: 4px solid #3498db; margin-bottom: 20px;">
                <p style="color: #34495e; margin: 0;">{formatted_answer}</p>
            </div>
        </div>
    ''' 