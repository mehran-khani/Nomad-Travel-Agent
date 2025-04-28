"""Main entry point for the Nomad Travel Agent."""

import argparse
import pathlib
import sys
import tempfile
import traceback

# --- Add imports for browser opening ---
import webbrowser

import google.genai as genai
from langchain_core.messages import HumanMessage

from . import (
    config,
    data_processing,
    display,
    phase1_logic,
    phase2_logic,
    schemas,
    vector_store,
)

# --- End added imports ---


def run_agent(interactive: bool):
    """Main function to run the Nomad travel agent."""
    print("--- 🚀 Starting Nomad Travel Agent ---")
    config.INTERACTIVE_MODE = interactive
    print(f"Mode: {'Interactive' if interactive else 'Demo'}")

    # 1. Configure Google API Client (Using config)
    genai_client = None
    if config.GOOGLE_API_KEY:
        try:
            genai_client = genai.Client(api_key=config.GOOGLE_API_KEY)
            print("✅ Google GenAI Client Configured.")
        except Exception as e:
            print(f"❌ ERROR: Failed to configure Google GenAI Client: {e}")
            print("   Please check your GOOGLE_API_KEY.")
            sys.exit(1)
    else:
        print("❌ ERROR: GOOGLE_API_KEY not found or loaded.")
        print("   The agent requires this key to function.")
        sys.exit(1)

    # 2. Load and Process Data
    df_processed, unique_cities = data_processing.load_and_process_wikivoyage_data()
    if df_processed is None or not unique_cities:
        print("❌ ERROR: Failed to load or process Wikivoyage data. Exiting.")
        sys.exit(1)
    unique_cities_lower_set = {city.lower() for city in unique_cities}

    # 3. Setup Vector Store
    poi_collection, gemini_embedder_query = vector_store.setup_chroma_db(
        df_processed, genai_client  # Pass the initialized client
    )
    if poi_collection is None or gemini_embedder_query is None:
        print("❌ ERROR: Failed to setup ChromaDB Vector Store. Exiting.")
        sys.exit(1)

    # 4. Build Phase 1 Graph
    try:
        # Pass the client if phase1 logic needs it for LLM initialization
        suggestion_graph = phase1_logic.build_suggestion_graph(unique_cities_lower_set)
    except Exception as e:
        print(f"❌ ERROR: Failed to build Phase 1 graph: {e}")
        traceback.print_exc()  # Print traceback for build errors
        sys.exit(1)

    # 5. Run Phase 1
    print("\n--- Starting Phase 1: Preference Gathering & Recommendations ---")
    initial_state = schemas.SuggestionState(
        messages=[HumanMessage(content="Hi, I'd like to plan a trip.")],
        user_preferences={},
        text_recommendations=None,
        recommendations=[],
        cities_with_data=[],
        error_message=None,
        is_finished=False,
        selected_city_for_phase_2=None,
    )

    # Reset demo index if needed
    if not interactive:
        phase1_logic.demo_answer_index = 0
        print("🔄 Demo answer index reset.")

    final_state_phase1 = None
    try:
        final_state_phase1 = suggestion_graph.invoke(
            initial_state, config={"recursion_limit": config.RECURSION_LIMIT}
        )
        print("\n--- Phase 1 Graph Execution Complete ---")

        if final_state_phase1:
            print("Final State Summary:")
            selected_city_for_phase2 = final_state_phase1.get(
                "selected_city_for_phase_2"
            )
            if selected_city_for_phase2:
                print(
                    f"  >>> Selected City for Phase 2: {selected_city_for_phase2} <<<"
                )
            else:
                print("  >>> No city selected for Phase 2. <<<")
        else:
            print("Phase 1 did not return a final state.")

    except Exception as e:
        print(f"\n--- ❌ An error occurred during Phase 1 graph execution: {e} ---")
        traceback.print_exc()
        sys.exit(1)

    # 6. Execute Phase 2 (if city selected)
    selected_city = (
        final_state_phase1.get("selected_city_for_phase_2")
        if final_state_phase1
        else None
    )
    if selected_city:
        print(f"\n--- Proceeding to Phase 2 for: {selected_city} ---")

        # Initialize Phase 2 components (pass the client)
        if not phase2_logic.initialize_phase2_components(genai_client):
            print("❌ ERROR: Failed to initialize Phase 2 components. Cannot proceed.")
            sys.exit(1)

        user_prefs = final_state_phase1.get("user_preferences", {})
        # Find country and image URL from Phase 1 recommendations
        selected_city_country = None
        selected_city_img_url = config.PLACEHOLDER_IMAGE_URL
        phase1_recs = final_state_phase1.get("recommendations", [])
        for rec in phase1_recs:
            if rec.get("city") == selected_city:
                selected_city_country = rec.get("country")
                selected_city_img_url = rec.get(
                    "image_url", config.PLACEHOLDER_IMAGE_URL
                )
                break

        city_info_dict = None
        try:
            # Execute Phase 2 logic
            city_info_dict = phase2_logic.execute_phase2(
                selected_city=selected_city,
                preferences=user_prefs,
                country=selected_city_country,
                collection=poi_collection,
                query_embedder=gemini_embedder_query,
            )
        except Exception as e:
            print(f"❌ Unhandled error during Phase 2 execution: {e}")
            traceback.print_exc()

        if city_info_dict:
            print("✅ Phase 2 execution successful.")

            # --- Generate, Save, and Open HTML ---
            try:
                print("   Generating HTML details...")
                html_city_details = display.display_city_details_html(
                    city_info_dict, selected_city_img_url
                )
                print("   HTML details generated.")

                # Create a temporary file that persists after closing
                # Use os.path.join for cross-platform compatibility if needed
                with tempfile.NamedTemporaryFile(
                    suffix=".html", mode="w", encoding="utf-8", delete=False
                ) as temp_file:
                    temp_file.write(html_city_details)
                    temp_file.flush()  # Ensure it's written to disk
                    file_path = temp_file.name
                    print(f"   Saved details to temporary file: {file_path}")

                # Convert path to file URI (e.g., file:///...)
                file_url = pathlib.Path(file_path).as_uri()

                print(f"   Attempting to open in browser: {file_url}")
                opened = webbrowser.open(file_url)  # Open the URL

                if not opened:
                    print(
                        "   ⚠️ Failed to automatically open browser. Please open the file manually:"
                    )
                    print(f"      {file_path}")  # Print path again for easy copy/paste
                else:
                    print("   ✅ Browser open request sent.")
                    # Optional: pause briefly if browser needs time before Q&A starts
                    # time.sleep(2)

            except Exception as html_err:
                print(f"❌ Error generating or opening HTML details: {html_err}")
                traceback.print_exc()
            # --- End Generate, Save, and Open HTML ---

            # 7. Run Q&A Session (regardless of HTML opening success)
            phase2_logic.run_qna_session(
                selected_city=selected_city,
                city_info_dict=city_info_dict,
                collection=poi_collection,
                query_embedder=gemini_embedder_query,
            )
        else:
            print(
                f"⚠️ Phase 2 failed to retrieve detailed information for {selected_city}."
            )

    else:
        print("\n--- Phase 1 ended without a city selection. Skipping Phase 2. ---")

    print("\n--- Nomad Travel Agent Finished ---")


def main():
    """Entry point for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Run the Nomad Conversational Travel Agent."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompts for user input).",
    )
    args = parser.parse_args()
    run_agent(interactive=args.interactive)


if __name__ == "__main__":
    main()
