# Nomad Travel Agent

An intelligent travel recommendation system that helps users discover and plan their perfect trip using advanced AI and natural language processing.

## Features & Capabilities

- Interactive travel preference gathering through natural conversation
- Smart city recommendations based on user preferences
- Detailed city information including points of interest and events
- Real-time weather data integration
- Image suggestions for destinations
- Interactive Q&A about recommended destinations

## Technologies Used

- Google Generative AI (Gemini)
- LangGraph for conversation flow
- LangChain for LLM orchestration
- ChromaDB for vector storage and retrieval
- OpenWeatherMap API for weather data
- Unsplash API for destination images
- Wikivoyage data for travel information

## Project Structure

```
nomad-travel-agent/
├── .github/workflows/      # CI configuration
├── data/                   # Data storage
├── notebooks/             # Jupyter notebooks
├── src/nomad_agent/      # Main package
│   ├── config.py         # Configuration and constants
│   ├── data_processing.py # Data loading and processing
│   ├── display.py        # HTML display functions
│   ├── main.py          # Entry point
│   ├── phase1_graph.py  # Preference gathering
│   ├── phase2_logic.py  # City details and Q&A
│   ├── schemas.py       # Data models
│   ├── tools.py         # External API integrations
│   └── vector_store.py  # ChromaDB setup
└── scripts/              # Utility scripts
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone 
   cd nomad-travel-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   # or
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API keys for:
   - Google AI (Gemini)
   - OpenWeatherMap
   - Unsplash

## How to Run

### Interactive Mode
```bash
python -m src.nomad_agent.main --interactive
```

### Demo Mode
```bash
python -m src.nomad_agent.main
```

## Key Components

- **Phase 1 (phase1_graph.py)**: Handles user preference gathering and initial city recommendations using a LangGraph-based conversation flow.
- **Phase 2 (phase2_logic.py)**: Provides detailed city information and handles the Q&A session using RAG (Retrieval-Augmented Generation).
- **Vector Store (vector_store.py)**: Manages the ChromaDB setup for efficient travel information retrieval.
- **Tools (tools.py)**: Integrates external APIs for weather data and images.

## Limitations & Future Work

- Currently limited to English language support
- Weather data is current only, no historical or forecast data
- Image suggestions could be more context-aware
- Could benefit from more structured activity recommendations
- Potential for multi-city trip planning

## Acknowledgements

- Wikivoyage for the comprehensive travel data
- OpenWeatherMap for weather information
- Unsplash for destination images
- Google for the Gemini AI platform

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 