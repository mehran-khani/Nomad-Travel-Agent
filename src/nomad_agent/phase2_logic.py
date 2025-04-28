import json
import traceback
from typing import Any, Dict, List, Optional

import chromadb
import google.api_core.exceptions
import google.genai as genai
from chromadb.api.types import EmbeddingFunction
from google.genai import types as genai_types
from pydantic import ValidationError

from . import config, schemas, tools, vector_store

phase2_llm_client: Optional[genai.client.Client] = None
phase2_initialized = False


def initialize_phase2_components(
    client: Optional[genai.client.Client],
):
    """Stores the GenAI client needed for Phase 2 operations."""
    global phase2_llm_client, phase2_initialized
    if phase2_initialized and phase2_llm_client is not None:
        print("ℹ️ Phase 2 components already initialized.")
        return True
    if client is None:
        print(
            "❌ ERROR (phase2_logic): GenAI client not provided. Phase 2 cannot proceed."
        )
        phase2_initialized = False
        return False
    phase2_llm_client = client
    phase2_initialized = True
    print("✅ Phase 2 components ready (using provided GenAI client).")
    return True


def get_phase2_system_prompt() -> str:
    """Generates the system prompt for the Phase 2 LLM, including the output schema."""
    try:

        if hasattr(schemas.CityInformation, "model_json_schema"):
            schema_dict = schemas.CityInformation.model_json_schema()
        elif hasattr(schemas.CityInformation, "schema"):
            schema_dict = schemas.CityInformation.schema()
        else:
            raise TypeError(
                "Schema class does not have a recognized schema generation method."
            )

        schema_dict_modified = schema_dict.copy()
        if "required" in schema_dict_modified and isinstance(
            schema_dict_modified["required"], list
        ):
            schema_dict_modified["required"] = [
                req
                for req in schema_dict_modified["required"]
                if req != "weather_summary"
            ]
        elif (
            "properties" in schema_dict_modified
            and "weather_summary" in schema_dict_modified["properties"]
        ):

            pass

        phase2_output_schema_str = json.dumps(schema_dict_modified, indent=2)

        prompt = f"""You are a specialized travel information assistant. Your goal is to provide a detailed, helpful, and engaging overview of a selected city based on user preferences, retrieved local Points of Interest (POIs), and recent events found via grounding.

**User Input:**
You will receive:
1. The selected city name.
2. A summary of the user's preferences (vibe, activities, weather, budget).
3. Context containing relevant Points of Interest (POIs) retrieved from a database for the selected city.

**Your Tasks:**

1.  **Events:** Use your grounding capabilities (Google Search) to find 2-4 relevant current or upcoming notable events (like festivals, major concerts, exhibitions, significant local happenings) in the `selected city`. Do NOT just list generic activities.
2.  **RAG POI Selection:** Analyze the provided `Context POIs`. Select 3-5 diverse POIs that BEST match the user's `preferences`. Prioritize POIs mentioned in the context.
3.  **Generate Content:** Synthesize the gathered information (preferences, RAG POIs, grounded events) into a structured JSON output. **DO NOT include weather information in your response.**

**Output Requirements:**

*   **Strict JSON:** You MUST output ONLY a single, valid JSON object conforming exactly to the following schema, **excluding the 'weather_summary' field**. Do NOT include any introductory text, explanations, apologies, or closing remarks outside the JSON structure.
*   **Schema (Target Structure - You will omit weather_summary):**
    ```json
    {phase2_output_schema_str}
    ```
*   **Field Details:**
    *   `city_name`, `country_name`: Fill accurately based on input.
    *   `general_summary`: Write a compelling 2-4 sentence paragraph introducing the city, connecting it to the user's general `vibe` and `preferences`.
    *   `weather_summary`: **<<< OMIT THIS FIELD FROM YOUR JSON OUTPUT >>>**
    *   `points_of_interest`: Populate the list using the POIs you selected from the `RAG context` that match user `preferences`. Ensure the `name`, `type`, and `description` fields are filled based *only* on the provided RAG context. The description should briefly mention why it fits the preferences.
    *   `events`: Populate the list using the current/upcoming events found via `grounding`. Provide a concise name and summary for each.

**Example Internal Thought Process:**
1. Receive Input: City="Paris", Prefs={{'vibe':'romantic', 'activities':'museums', 'budget':'mid-range'}}, RAG Context=[POI data...].
2. Tool Call Planning: Need to use grounding for events. *No weather tool call needed*.
3. Generate Final Response (using Grounding internally for events): Construct the JSON output (WITHOUT weather_summary), selecting relevant museums from RAG context for the POI list, finding events like "Louvre Late Nights" via grounding for the events list, and writing the summary paragraphs.
"""
        return prompt
    except Exception as e:
        print(f"❌ ERROR creating Phase 2 system prompt: {e}")
        traceback.print_exc()

        return "ERROR: System prompt generation failed. Please check logs."


def execute_phase2(
    selected_city: str,
    preferences: Dict[str, Any],
    country: Optional[str],
    collection: Optional[chromadb.Collection],
    query_embedder: Optional[EmbeddingFunction],
) -> Optional[Dict[str, Any]]:
    """Orchestrates Phase 2: RAG, Weather, Grounded LLM call, Parsing."""
    if not phase2_initialized or phase2_llm_client is None:
        print("❌ Phase 2 Error: LLM client not initialized.")
        return None
    if not collection or not query_embedder:
        print("❌ Phase 2 Error: ChromaDB collection or query embedder not available.")
        return None

    print(f"\n--- Starting Phase 2 Execution for {selected_city} ---")

    print("Step 1/6: Retrieving POIs via RAG...")

    rag_docs: List[str] = []
    try:
        rag_docs = vector_store.retrieve_rag_documents(
            city=selected_city,
            preferences=preferences,
            collection=collection,
            embedder=query_embedder,
            n_results=config.RAG_N_RESULTS,
        )
        if rag_docs:
            rag_context = "\n\n".join(
                [f"Context POI {i+1}:\n{doc}" for i, doc in enumerate(rag_docs)]
            )
            print(f"   Retrieved {len(rag_docs)} RAG documents.")
        else:
            rag_context = "No relevant Points of Interest found in the database for this city and preferences."
            print("   Phase 2 RAG: No documents retrieved.")
    except Exception as rag_err:
        print(f"❌ Phase 2 RAG: Error during retrieval: {rag_err}")
        traceback.print_exc()
        rag_context = "Error retrieving Points of Interest from the database."

    print("Step 2/6: Fetching weather data...")
    weather_details_dict = tools.get_placeholder_weather(selected_city, country)
    weather_tool = tools.phase2_direct_tools.get("get_weather")

    if weather_tool and callable(getattr(weather_tool, "invoke", None)):
        try:
            weather_input = {"city": selected_city}
            if country:
                weather_input["country"] = country
            weather_details_dict = weather_tool.invoke(weather_input)
            if isinstance(weather_details_dict, dict):
                print(
                    f"   Weather tool returned: {json.dumps(weather_details_dict, indent=2)}"
                )
            else:
                print(
                    f"   Warning: Weather tool did not return a dictionary: {weather_details_dict}"
                )
                weather_details_dict = tools.get_placeholder_weather(
                    selected_city, country
                )

        except Exception as weather_e:
            print(f"❌ Phase 2 Weather: Error calling get_weather tool: {weather_e}")
            traceback.print_exc()
    else:
        print("   Phase 2 Weather Error: get_weather tool not found or not invokable.")

    print("Step 3/6: Constructing LLM prompt...")
    phase2_system_prompt = get_phase2_system_prompt()
    if "ERROR" in phase2_system_prompt:
        print("❌ Phase 2 Error: Could not generate system prompt.")
        return None

    pref_summary = ", ".join([f"{k}: {v}" for k, v in preferences.items() if v])
    user_request_content = f"""
Selected City: {selected_city} {f"({country})" if country else ""}
User Preferences Summary: [{pref_summary}]

Retrieved Points of Interest Context:
---
{rag_context}
---

Please generate the detailed city information (excluding weather) based on all instructions in the system prompt. Output only the valid JSON object.
"""

    prompt_history = [
        genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(
                    text=phase2_system_prompt
                    + "\n\nUSER REQUEST:\n"
                    + user_request_content
                )
            ],
        )
    ]

    grounding_tools_list = None
    try:

        search_tool_for_api = genai_types.Tool(google_search=genai_types.GoogleSearch())
        grounding_tools_list = [search_tool_for_api]
        print("   Grounding tool configured for LLM call.")
    except AttributeError:
        print(
            "⚠️ Warning: google.generativeai.types.GoogleSearch not found. Grounding might fail. Check library version."
        )
    except Exception as tool_err:
        print(
            f"⚠️ Warning: Error configuring grounding tool: {tool_err}. Proceeding without grounding."
        )

    print("Step 4/6: Calling Phase 2 LLM...")
    generated_text = None
    try:
        content_config_with_grounding = genai_types.GenerateContentConfig(
            tools=grounding_tools_list
        )

        response = phase2_llm_client.models.generate_content(
            model=config.PHASE2_MODEL_NAME,
            contents=prompt_history,
            config=content_config_with_grounding,
        )
        print("   LLM call finished.")

        print("Step 5/6: Processing LLM response...")

        if not response.candidates:
            print("❌ Phase 2 Error: LLM response missing candidates.")
            print(f"   Full response prompt feedback: {response.prompt_feedback}")
            return None

        candidate = response.candidates[0]
        if not (candidate.content and candidate.content.parts):
            print("❌ Phase 2 Error: LLM response candidate missing content or parts.")
            print(f"   Finish Reason: {candidate.finish_reason}")
            print(f"   Full response prompt feedback: {response.prompt_feedback}")
            return None

        generated_text = candidate.content.parts[0].text
        print("   LLM response text extracted.")

    except google.api_core.exceptions.GoogleAPIError as api_error:
        print(
            f"❌ Phase 2 Error: Google API Error during generate_content: {api_error}"
        )
        return None
    except Exception as llm_call_err:
        print(
            f"❌ An unexpected error occurred during Phase 2 LLM call: {type(llm_call_err).__name__} - {llm_call_err}"
        )
        traceback.print_exc()
        return None

    print("Step 6/6: Parsing response, adding weather, validating...")
    if generated_text:

        parsed_data = None
        json_string_to_parse = None
        try:
            text_to_parse = generated_text.strip()
            print(f"   DEBUG: Raw text length: {len(text_to_parse)}")

            if text_to_parse.startswith("```json"):
                text_to_parse = text_to_parse[7:]
            elif text_to_parse.startswith("```"):
                text_to_parse = text_to_parse[3:]

            if text_to_parse.endswith("```"):
                text_to_parse = text_to_parse[:-3]

            text_to_parse = text_to_parse.strip()

            first_brace_index = text_to_parse.find("{")
            last_closing_brace_index = text_to_parse.rfind("}")

            if (
                first_brace_index != -1
                and last_closing_brace_index != -1
                and last_closing_brace_index > first_brace_index
            ):

                json_string_to_parse = text_to_parse[
                    first_brace_index : last_closing_brace_index + 1
                ]
                print(
                    f"   DEBUG: Extracted JSON candidate using first/last brace after cleaning (Length: {len(json_string_to_parse)})."
                )
            else:

                print(
                    "   DEBUG: Could not find valid JSON start/end braces '{...}' after cleaning markdown."
                )

                json_string_to_parse = text_to_parse
                if not json_string_to_parse.startswith(
                    "{"
                ) or not json_string_to_parse.endswith("}"):
                    print(
                        "   WARNING: Attempting to parse string that doesn't start/end with braces."
                    )

            if not json_string_to_parse:
                raise ValueError(
                    "Could not extract a JSON candidate string after cleaning."
                )

            print(
                f"   DEBUG: Attempting to parse JSON string (length {len(json_string_to_parse)}): {json_string_to_parse[:100]}..."
            )
            parsed_data = json.loads(json_string_to_parse)
            print("   DEBUG: JSON parsing successful.")

            if isinstance(parsed_data, dict):
                parsed_data["weather_details"] = weather_details_dict
                print("   DEBUG: Added weather details.")
            else:
                print(
                    "⚠️ Warning: Parsed data was not a dictionary. Cannot add weather."
                )

                return None

            try:

                temp_weather = parsed_data.pop("weather_details", None)

                print(
                    "✅ Phase 2 main data parsed and validated against Pydantic schema."
                )

                if temp_weather:
                    parsed_data["weather_details"] = temp_weather
                return parsed_data

            except ValidationError as pydantic_error:
                print(
                    f"⚠️ Phase 2 Warning: Parsed data failed Pydantic validation: {pydantic_error}"
                )

                if temp_weather:
                    parsed_data["weather_details"] = temp_weather
                print(
                    f"   Data that failed validation (with weather added back): {json.dumps(parsed_data, indent=2)}"
                )

                return parsed_data
            except Exception as validation_err:
                print(f"❌ Phase 2 Error during Pydantic validation: {validation_err}")
                traceback.print_exc()

                if temp_weather:
                    parsed_data["weather_details"] = temp_weather
                return None

        except json.JSONDecodeError as json_error:
            print(
                f"❌ Phase 2 Error: Failed to decode LLM response as JSON: {json_error}"
            )
            log_string = (
                json_string_to_parse
                if json_string_to_parse is not None
                else text_to_parse
            )
            print(f"   Attempted to parse: '{log_string[:500]}...'")
            print(f"   LLM Raw Output:\n---\n{generated_text}\n---")
            return None
        except ValueError as val_err:
            print(f"❌ Phase 2 Error: {val_err}")
            print(f"   LLM Raw Output:\n---\n{generated_text}\n---")
            return None
        except Exception as parse_validate_error:
            print(
                f"❌ Phase 2 Error: Unexpected error during JSON processing or validation: {parse_validate_error}"
            )
            traceback.print_exc()
            return None
    else:
        print("❌ Phase 2 Error: No text generated by the LLM (final check).")
        return None


def run_qna_session(
    selected_city: str,
    city_info_dict: Dict[str, Any],
    collection: Optional[chromadb.Collection],
    query_embedder: Optional[EmbeddingFunction],
):
    """Runs the interactive or demo Q&A session for the selected city."""

    if not phase2_initialized or phase2_llm_client is None:
        print("❌ Q&A Error: LLM Client not available.")
        return
    if not collection or not query_embedder:
        print("❌ Q&A Error: ChromaDB collection or query embedder not available.")
        return

    print(f"\n--- Starting Q&A for: {selected_city} ---")

    points_of_interest = city_info_dict.get("points_of_interest", [])
    events = city_info_dict.get("events", [])
    displayed_pois_text = "None initially listed."
    if points_of_interest and isinstance(points_of_interest, list):
        poi_lines = []
        for i, poi in enumerate(points_of_interest):
            if isinstance(poi, dict):
                poi_lines.append(
                    f"{i+1}. Name: {poi.get('name', 'N/A')}, Type: {poi.get('type', 'N/A')}, Desc: {poi.get('description', 'N/A')}"
                )
        if poi_lines:
            displayed_pois_text = (
                "\nPreviously suggested Points of Interest:\n" + "\n".join(poi_lines)
            )

    initial_events_text = "None initially listed."
    if events and isinstance(events, list):
        event_lines = []
        for ev in events:
            if isinstance(ev, dict):
                event_lines.append(
                    f"- {ev.get('name', 'N/A')}: {ev.get('summary', 'N/A')}"
                )
        if event_lines:
            initial_events_text = (
                "\nSome recent/upcoming events mentioned earlier:\n"
                + "\n".join(event_lines)
            )

    qna_model_name = config.QNA_MODEL_NAME

    if config.INTERACTIVE_MODE:
        print("Feel free to ask about the places mentioned above or other details.")
        print("Type 'quit' or 'exit' to end the chat.")
        while True:
            try:
                user_question = input(f"\n👤 Ask about {selected_city}: ")
                user_question_clean = user_question.strip().lower()
                if user_question_clean in ["quit", "exit", "q", "bye"]:
                    print("\nEnding Q&A session. Safe travels!")
                    break
                if not user_question.strip():
                    continue

                print("🤖 Nomad is searching for an answer...")

                rag_query_prefs = {"user_question": user_question}
                rag_context_docs = vector_store.retrieve_rag_documents(
                    city=selected_city,
                    preferences=rag_query_prefs,
                    collection=collection,
                    embedder=query_embedder,
                    n_results=3,
                )
                rag_context_for_llm = (
                    "\n---\n".join(rag_context_docs)
                    if rag_context_docs
                    else "No specific details found in the local database regarding that topic."
                )

                qna_prompt_text = f"""You are Nomad, a friendly travel assistant discussing {selected_city} with a user. Use a helpful and informative tone, avoiding repetitive greetings.

Here's some information that was initially presented to the user:
{displayed_pois_text}
{initial_events_text}
(Note: The initial event list might include upcoming or recurring events, not necessarily live ones).

Now, answer the user's latest question based on the information above AND the relevant snippets retrieved below.

Instructions:
1.  **Acknowledge the Question:** Briefly acknowledge the user's topic (e.g., "Regarding [topic]...").
2.  **Prioritize Specifics:** If the user asks about a *specific POI or event mentioned above or in the snippets*, focus on providing details found in *either* the initial context OR the retrieved snippets. Combine information if possible.
3.  **General Event Questions:** If asked *generally* about "current events" or "what's happening now", check the initial event list and snippets. If no *clearly current* events are found (besides daily activities like Djemaa El-Fna), explain that the available information focuses on known points of interest and some specific listed events which might not be live, and recommend checking local resources for real-time updates.
4.  **Insufficient Info:** If neither the initial context nor the snippets contain relevant information to answer the question, state that you don't have those specific details in the provided context.
5.  **Conciseness:** Keep answers concise (2-4 sentences typically) and relevant to the question. Avoid making up information.

Retrieved Context Snippets (Primarily for POIs/General Info, but might mention events):
---
{rag_context_for_llm}
---

User Question: {user_question}

Nomad's Answer:"""
                grounding_tools_list = None
                try:

                    search_tool_for_api = genai_types.Tool(
                        google_search=genai_types.GoogleSearch()
                    )
                    grounding_tools_list = [search_tool_for_api]
                    print("   Grounding tool configured for LLM call.")
                except AttributeError:
                    print(
                        "⚠️ Warning: google.generativeai.types.GoogleSearch not found. Grounding might fail. Check library version."
                    )
                except Exception as tool_err:
                    print(
                        f"⚠️ Warning: Error configuring grounding tool: {tool_err}. Proceeding without grounding."
                    )
                content_config_with_grounding = genai_types.GenerateContentConfig(
                    tools=grounding_tools_list
                )

                qna_response = phase2_llm_client.models.generate_content(
                    model=qna_model_name,
                    contents=[
                        genai_types.Content(
                            role="user", parts=[genai_types.Part(text=qna_prompt_text)]
                        )
                    ],
                    config=content_config_with_grounding,
                )

                llm_answer_text = "Sorry, I had trouble formulating an answer to that."
                if qna_response.candidates and qna_response.candidates[0].content.parts:
                    llm_answer_text = qna_response.candidates[0].content.parts[0].text

                print(f"\n🤖 Nomad:\n{llm_answer_text}")

            except google.api_core.exceptions.GoogleAPIError as api_err:
                print(f"\n❌ API Error during Q&A: {api_err}")
                print(
                    "   Sorry, I encountered an issue connecting to my knowledge base."
                )
            except Exception as qna_err:
                print("\n--- ❌ Error during Q&A ---")
                print(f"Error Type: {type(qna_err).__name__}")
                print(f"Error Details: {qna_err}")
                traceback.print_exc()
                print(
                    "Sorry, I encountered a problem processing that question. Please try phrasing it differently or type 'quit'."
                )

    else:
        print("🤖 Demo Mode: Running predefined Q&A questions...")
        demo_questions = []

        if (
            points_of_interest
            and isinstance(points_of_interest, list)
            and len(points_of_interest) > 0
            and isinstance(points_of_interest[0], dict)
        ):
            first_poi_name = points_of_interest[0].get(
                "name", "the first point of interest"
            )
            demo_questions.append(f"Tell me more about {first_poi_name}.")
        else:
            demo_questions.append("What is there to see there?")

        if (
            events
            and isinstance(events, list)
            and len(events) > 0
            and isinstance(events[0], dict)
        ):
            first_event_name = events[0].get("name", "the first event")
            demo_questions.append(f"What is the {first_event_name} about?")
        else:
            demo_questions.append("Any interesting food recommendations?")

        question_count = 0
        MAX_DEMO_QUESTIONS = config.MAX_DEMO_QUESTIONS

        for user_question in demo_questions:
            if question_count >= MAX_DEMO_QUESTIONS:
                break
            print(f"\n👤 Demo Question: {user_question}")
            try:
                print("🤖 Nomad is searching for an answer...")

                rag_query_prefs = {"user_question": user_question}
                rag_context_docs = vector_store.retrieve_rag_documents(
                    city=selected_city,
                    preferences=rag_query_prefs,
                    collection=collection,
                    embedder=query_embedder,
                    n_results=3,
                )
                rag_context_for_llm = (
                    "\n---\n".join(rag_context_docs)
                    if rag_context_docs
                    else "No specific details found in the local database regarding that topic."
                )

                qna_prompt_text = f"""You are Nomad, a friendly travel assistant discussing {selected_city}...
                (rest of prompt identical to interactive version)
                ...
                User Question: {user_question}

                Nomad's Answer:"""

                grounding_tools_list = None
                try:

                    search_tool_for_api = genai_types.Tool(
                        google_search=genai_types.GoogleSearch()
                    )
                    grounding_tools_list = [search_tool_for_api]
                    print("   Grounding tool configured for LLM call.")
                except AttributeError:
                    print(
                        "⚠️ Warning: google.generativeai.types.GoogleSearch not found. Grounding might fail. Check library version."
                    )
                except Exception as tool_err:
                    print(
                        f"⚠️ Warning: Error configuring grounding tool: {tool_err}. Proceeding without grounding."
                    )
                content_config_with_grounding = genai_types.GenerateContentConfig(
                    tools=grounding_tools_list
                )

                qna_response = phase2_llm_client.models.generate_content(
                    model=qna_model_name,
                    contents=[
                        genai_types.Content(
                            role="user", parts=[genai_types.Part(text=qna_prompt_text)]
                        )
                    ],
                    config=content_config_with_grounding,
                )

                llm_answer_text = "Sorry, I had trouble formulating an answer to that."
                if qna_response.candidates and qna_response.candidates[0].content.parts:
                    llm_answer_text = qna_response.candidates[0].content.parts[0].text
                print(f"\n🤖 Nomad:\n{llm_answer_text}")

                question_count += 1

            except google.api_core.exceptions.GoogleAPIError as api_err:
                print(
                    f"\n❌ API Error during Demo Q&A for question '{user_question}': {api_err}"
                )
                print("   Skipping this demo question.")
            except Exception as qna_err:
                print(
                    f"\n--- ❌ Error during Demo Q&A for question: '{user_question}' ---"
                )
                print(f"Error Type: {type(qna_err).__name__}")
                print(f"Error Details: {qna_err}")
                print("   Skipping this demo question due to error.")
                traceback.print_exc()

        print("\n🏁 Demo Mode: Finished predefined Q&A questions.")

    print("--- Q&A Session Finished ---")
