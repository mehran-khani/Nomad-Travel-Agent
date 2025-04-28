import sys
import traceback
from functools import partial
from typing import Any, Dict, Literal, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from . import config, schemas, tools

suggestion_llm = None
suggestion_llm_with_tools = None
parser_llm = None
structured_parser_llm = None
recommender_llm = None
structured_recommender_llm = None
llms_initialized = False


def initialize_phase1_llms():
    """Initializes LLMs needed for Phase 1. Returns True on success, False otherwise."""
    global suggestion_llm, suggestion_llm_with_tools, parser_llm, structured_parser_llm, recommender_llm, structured_recommender_llm, llms_initialized
    if llms_initialized:
        return True
    if not config.GOOGLE_API_KEY:
        print(
            "❌ ERROR (phase1_graph): Google API Key not found. Phase 1 LLMs cannot function."
        )
        return False

    print("⚙️ Initializing Phase 1 LLMs...")
    try:
        common_kwargs = {
            "model": config.SUGGESTION_AGENT_MODEL,
            "google_api_key": config.GOOGLE_API_KEY,
        }

        suggestion_llm = ChatGoogleGenerativeAI(**common_kwargs, temperature=0.7)
        suggestion_llm_with_tools = suggestion_llm.bind_tools(tools.suggestion_tools)

        parser_llm = ChatGoogleGenerativeAI(**common_kwargs, temperature=0.0)
        structured_parser_llm = parser_llm.with_structured_output(
            schemas.ParsedPreference
        )

        recommender_llm = ChatGoogleGenerativeAI(**common_kwargs, temperature=0.7)
        structured_recommender_llm = recommender_llm.with_structured_output(
            schemas.CityRecommendationsList
        )

        print("✅ Phase 1 LLMs initialized successfully.")
        llms_initialized = True
        return True
    except Exception as e:
        print(f"❌ ERROR initializing Phase 1 LLMs: {e}")
        traceback.print_exc()
        return False


ASK_QUESTION_SYSTEM_PROMPT = """You are 'Nomad,' a friendly AI travel assistant. Your current goal is ONLY to ask the *next* relevant question to understand the user's travel preferences.
Analyze the conversation history. Identify which core preferences ('vibe', 'activities', 'weather', 'budget') have ALREADY been discussed.
Ask ONE clear, concise question about the NEXT logical preference category that has NOT been discussed yet.
Start with 'vibe', then 'activities', then 'weather', then 'budget'.
**When asking a question, provide a few diverse examples in parentheses to help the user.**
If the user provides information about multiple preferences at once, acknowledge it briefly and ask about the next *required* category they *didn't* cover.
DO NOT suggest destinations. DO NOT ask for information you already have. Keep the conversation flowing naturally. Start the very first turn by asking about the 'vibe'.

Example History (Vibe asked):
[...]
User: I want a relaxing beach vacation.
You: Sounds lovely! What kind of activities do you enjoy on a relaxing trip? (e.g., spa treatments, reading by the pool, gentle walks, exploring local cafes?)

Example History (Activities asked):
[...]
User: Hiking sounds fun. I prefer cool weather.
You: Hiking in cool weather, got it! And what's your general budget looking like for this trip? (e.g., budget-friendly, mid-range, luxury?)
"""
PREFERENCE_PARSING_SYSTEM_PROMPT = """Analyze the last user message in the context of the preceding agent question.
Extract the preference category ('vibe', 'activities', 'weather', 'budget', or null) and the user's stated preference value.
Output ONLY JSON matching the 'ParsedPreference' schema.
If the user's message doesn't clearly state a preference for the asked category, set both preference_key and preference_value to null.
Agent Question: What kind of vibe are you hoping for?
User Response: I want something really relaxing and quiet.
JSON Output: {"preference_key": "vibe", "preference_value": "relaxing and quiet"}
Agent Question: What kind of weather do you prefer?
User Response: Doesn't matter much.
JSON Output: {"preference_key": "weather", "preference_value": null}"""

RECOMMENDATION_PROMPT_TEMPLATE = """
Based ONLY on the following user travel preferences:
Vibe: {vibe}
Activities: {activities}
Weather: {weather}
Budget Indication: {budget}

Recommend 3-5 diverse global cities that fit well.
For each city, provide: city name, country, a brief compelling description (~2-3 sentences), and a specific justification (~1-2 sentences) explaining the match.
Return ONLY the recommendations in the specified JSON format matching the 'CityRecommendationsList' schema. Do NOT add any introductory text or other commentary.
"""

demo_answer_index = 0


def ask_question_node(state: schemas.SuggestionState) -> Dict[str, Any]:
    """Invokes the LLM to ask the next preference question."""
    if not llms_initialized or suggestion_llm is None:
        print("❌ Error (ask_question): LLM not available.")
        return {
            "messages": [
                AIMessage(
                    content="My apologies, I'm having trouble thinking of the next question right now. Could you tell me about the vibe you're looking for?"
                )
            ],
            "error_message": "Question LLM not initialized.",
        }

    print("🤖 Nomad is thinking of the next question...")
    messages_for_llm = [SystemMessage(content=ASK_QUESTION_SYSTEM_PROMPT)] + state[
        "messages"
    ]

    try:
        ai_response: BaseMessage = suggestion_llm.invoke(messages_for_llm)
        if not isinstance(ai_response, AIMessage):
            ai_response = AIMessage(content=str(ai_response.content))
        return {"messages": [ai_response]}

    except Exception as e:
        print(f"❌ Error calling Question LLM: {type(e).__name__} - {e}")
        traceback.print_exc()
        return {
            "messages": [
                AIMessage(
                    content="I seem to be having trouble formulating my question. Could you perhaps tell me about the vibe you're looking for?"
                )
            ],
            "error_message": f"Failed to generate next question: {e}",
        }


def human_input_node(state: schemas.SuggestionState) -> Dict[str, Any]:
    """Gets input from the user or uses demo answers."""
    global demo_answer_index
    if state.get("is_finished"):
        print("Conversation marked as finished, skipping input.")
        return {}

    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, AIMessage):
        print(f"\n🤖 Nomad: {last_message.content}")
    else:
        print("Warning: Expected AIMessage before human input, proceeding anyway.")

    user_input = ""
    if config.INTERACTIVE_MODE:
        try:
            user_input = input("👤 You: ")
        except EOFError:
            print("\nInput stream closed, treating as quit.")
            user_input = "quit"
    else:
        if demo_answer_index < len(config.DEMO_PREFERENCE_ANSWERS):
            user_input = config.DEMO_PREFERENCE_ANSWERS[demo_answer_index]
            print(f"👤 You (Demo Input): {user_input}")
            demo_answer_index += 1
        else:
            print(
                "⚠️ Demo Mode: Ran out of predefined preference answers. Using 'quit'."
            )
            user_input = "quit"

    if user_input.strip().lower() in ["q", "quit", "exit", "bye"]:
        print("Exiting conversation loop.")
        return {"messages": [HumanMessage(content=user_input)], "is_finished": True}
    else:
        return {"messages": [HumanMessage(content=user_input)]}


def parse_preference_node(state: schemas.SuggestionState) -> Dict[str, Any]:
    """Parses the last user message and updates user_preferences."""
    if not llms_initialized or structured_parser_llm is None:
        print("❌ Error (parse_preference): Structured Parser LLM not available.")
        return {"error_message": "Parser LLM not initialized."}
    if state.get("is_finished"):
        return {}

    messages = state.get("messages", [])
    if len(messages) < 2:
        print("DEBUG (parse_preference): Not enough messages to parse.")
        return {}

    last_human_message = messages[-1]
    last_ai_message = messages[-2] if len(messages) >= 2 else None

    # Ensure we have the correct message types before proceeding
    if not isinstance(last_human_message, HumanMessage) or not isinstance(
        last_ai_message, AIMessage
    ):
        print("DEBUG (parse_preference): Incorrect message types for parsing.")
        return {}

    print("🤖 Nomad is parsing preference...")
    parsing_prompt_content = f"""Agent Question: {last_ai_message.content}
User Response: {last_human_message.content}"""

    messages_for_llm = [
        SystemMessage(content=PREFERENCE_PARSING_SYSTEM_PROMPT),
        HumanMessage(content=parsing_prompt_content),
    ]

    try:
        # Invoke the structured LLM
        parsed_result = structured_parser_llm.invoke(messages_for_llm)
        # Add extra debugging
        print(
            f"DEBUG (parse_preference): Raw parser result type: {type(parsed_result)}"
        )
        print(f"DEBUG (parse_preference): Raw parser result value: {parsed_result!r}")

        current_preferences = state.get("user_preferences", {}).copy()

        # --- REVISED CHECKING LOGIC ---
        if isinstance(parsed_result, schemas.ParsedPreference):
            # Access attributes - they should exist if it's a ParsedPreference instance,
            # potentially with value None due to Optional[str]
            key = parsed_result.preference_key
            value = parsed_result.preference_value

            # Explicitly check if BOTH key and value were successfully extracted (are not None)
            if key is not None and value is not None:
                if key in ["vibe", "activities", "weather", "budget"]:
                    print(f"   Parsed: {key} = '{value}'")
                    current_preferences[key] = value
                    # Return ONLY the updated preferences
                    return {"user_preferences": current_preferences}
                else:
                    # Log if the key is valid but not one we expect
                    print(f"   Parser returned unexpected key: {key}")
                    # Return empty update, do not modify state
                    return {}
            else:
                # Handle case where LLM returned the structure but couldn't fill key/value
                print(
                    f"   Parser found structure but key ({key}) or value ({value}) is None."
                )
                # Return empty update, do not modify state
                return {}
        else:
            # Handle case where the LLM output couldn't be parsed into ParsedPreference at all
            print(
                f"   Parser returned unexpected type: {type(parsed_result)}. Value: {parsed_result!r}"
            )
            # Return empty update, maybe add an error message if needed
            return {
                "error_message": f"Parser returned unexpected type: {type(parsed_result)}"
            }
        # --- END REVISED CHECKING LOGIC ---

    except Exception as e:
        print(f"❌ Error calling structured parser LLM: {type(e).__name__} - {e}")
        # Capture the exception type and message for better debugging
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_details = traceback.format_exception_only(exc_type, exc_value)
        print(f"   Error details: {''.join(error_details).strip()}")
        # Optionally print full traceback if needed for deeper debugging
        # traceback.print_exc()
        return {"error_message": f"Failed to parse user preference: {e}"}


def generate_recommendations_node(state: schemas.SuggestionState) -> Dict[str, Any]:
    """Generates text-only recommendations based on parsed preferences."""
    if not llms_initialized or structured_recommender_llm is None:
        print(
            "❌ Error (generate_recommendations): Structured Recommender LLM not available."
        )
        return {
            "error_message": "Recommender LLM not initialized.",
            "is_finished": True,
        }
    if state.get("is_finished"):
        return {}

    print("🤖 Nomad is generating recommendations...")
    prefs = state.get("user_preferences")
    required_keys = ["vibe", "activities", "weather", "budget"]

    if not prefs or not all(key in prefs for key in required_keys):
        missing = [key for key in required_keys if key not in prefs]
        err_msg = f"Cannot generate recommendations: Missing {', '.join(missing)} preference(s)."
        print(f"❌ Error: {err_msg}")

        return {
            "error_message": err_msg,
            "messages": [
                AIMessage(
                    content=f"Sorry, I still need to know about: {', '.join(missing)}."
                )
            ],
            "is_finished": True,
        }

    try:
        prompt_content = RECOMMENDATION_PROMPT_TEMPLATE.format(
            vibe=prefs.get("vibe", "not specified"),
            activities=prefs.get("activities", "not specified"),
            weather=prefs.get("weather", "not specified"),
            budget=prefs.get("budget", "not specified"),
        )
    except KeyError as e:
        print(f"❌ Error formatting recommendation prompt: Missing key {e}")
        return {"error_message": f"Prompt formatting error: {e}", "is_finished": True}

    messages_for_llm = [HumanMessage(content=prompt_content)]

    try:

        parsed_response: schemas.CityRecommendationsList = (
            structured_recommender_llm.invoke(messages_for_llm)
        )

        text_recs_list = []

        if (
            isinstance(parsed_response, schemas.CityRecommendationsList)
            and parsed_response.recommendations
        ):

            text_recs_list = [
                rec.model_dump() for rec in parsed_response.recommendations
            ]
            print(f"   Generated {len(text_recs_list)} recommendations.")

            return {"text_recommendations": text_recs_list}
        else:
            warn_msg = "Structured recommender returned invalid or empty response."
            print(f"⚠️ Warning: {warn_msg}")

            return {
                "error_message": warn_msg,
                "messages": [
                    AIMessage(
                        content="I had trouble generating recommendations in the right format. Let's stop here for now."
                    )
                ],
                "is_finished": True,
            }

    except Exception as e:
        print(f"❌ Error invoking structured recommender LLM: {type(e).__name__} - {e}")
        traceback.print_exc()

        return {
            "error_message": f"Failed to generate recommendations: {e}",
            "is_finished": True,
        }


def verify_recommendations_node(
    state: schemas.SuggestionState, unique_cities_lower: set
) -> Dict[str, Any]:
    """
    Checks generated text recommendations against the loaded dataset cities (case-insensitive).
    Adds 'has_data' flag to each recommendation in 'text_recommendations'.
    Populates 'cities_with_data' list.
    Requires unique_cities_lower set to be passed via functools.partial during graph setup.
    """
    if state.get("is_finished"):
        return {}
    print("Verifying recommendations against dataset...")

    text_recs = state.get("text_recommendations", [])
    if not text_recs:
        print("   Warning: No text recommendations found in state to verify.")
        return {"cities_with_data": []}

    if not unique_cities_lower:
        print("❌ Error: Cannot verify recommendations, city dataset list unavailable.")

        updated_text_recs = []
        for rec in text_recs:
            if isinstance(rec, dict):
                updated_rec = rec.copy()
                updated_rec["has_data"] = False
                updated_text_recs.append(updated_rec)
        return {
            "text_recommendations": updated_text_recs,
            "cities_with_data": [],
            "error_message": "Verification dataset missing.",
        }

    updated_text_recs = []
    cities_found_in_data = []

    for rec in text_recs:
        if not isinstance(rec, dict):
            print(f"   Warning: Skipping invalid item in text_recommendations: {rec}")
            continue

        city_name = rec.get("city", "")

        has_data = city_name.lower() in unique_cities_lower

        updated_rec = rec.copy()
        updated_rec["has_data"] = has_data
        updated_text_recs.append(updated_rec)

        if has_data:
            cities_found_in_data.append(city_name)

    print(f"   Verification complete. Found data for: {cities_found_in_data}")

    return {
        "text_recommendations": updated_text_recs,
        "cities_with_data": cities_found_in_data,
    }


def call_image_tool_node(state: schemas.SuggestionState) -> Dict[str, Any]:
    """Constructs the AIMessage with tool calls for Unsplash images."""
    if state.get("is_finished"):
        return {}
    print("Preparing image fetch requests...")

    text_recommendations = state.get("text_recommendations")

    if not text_recommendations:
        err_msg = "No text recommendations found to trigger image fetching."
        print(f"❌ Error: {err_msg}")

        return {
            "messages": [
                AIMessage(
                    content="Sorry, I couldn't find the recommendations list to get images for."
                )
            ],
            "error_message": err_msg,
            "is_finished": True,
        }

    tool_calls = []

    if not hasattr(tools, "UNSPLASH_TOOL_NAME"):
        err_msg = "Tool name configuration (UNSPLASH_TOOL_NAME) missing."
        print(f"❌ ERROR: {err_msg}")
        return {"error_message": err_msg, "is_finished": True}

    for i, rec in enumerate(text_recommendations):
        city = rec.get("city")
        country = rec.get("country")
        if city:

            call_id = f"{tools.UNSPLASH_TOOL_NAME}_{i}_{city.replace(' ', '_').lower()}"

            tool_calls.append(
                {
                    "name": tools.UNSPLASH_TOOL_NAME,
                    "args": {"city": city, "country": country},
                    "id": call_id,
                }
            )

    if not tool_calls:

        print("   Warning: No tool calls could be generated for images.")
        ai_message_content = "I have the recommendations, but couldn't prepare the image requests properly."

        return {"messages": [AIMessage(content=ai_message_content)]}

    ai_message_with_calls = AIMessage(
        content="Okay, I've generated recommendations. Let me quickly fetch some images...",
        tool_calls=tool_calls,
    )
    print(f"   Requesting {len(tool_calls)} image tool calls.")
    return {"messages": [ai_message_with_calls]}


def format_final_output_node(state: schemas.SuggestionState) -> Dict[str, Any]:
    """
    Formats the final response combining text recommendations and image URLs received
    from the ToolNode. Updates state['recommendations'].
    (Does not display HTML here, display is handled in main.py)
    """
    if state.get("is_finished"):
        return {}
    print("Formatting final recommendations...")

    text_recommendations = state.get("text_recommendations")
    messages = state.get("messages", [])

    if not text_recommendations:
        err_msg = "Cannot format final output, verified recommendations missing."
        print(f"❌ Error: {err_msg}")
        return {
            "messages": [
                AIMessage(
                    content="Sorry, I seem to have lost the recommendations list."
                )
            ],
            "error_message": err_msg,
            "is_finished": True,
        }
    if not hasattr(tools, "UNSPLASH_TOOL_NAME"):
        err_msg = "Tool name configuration (UNSPLASH_TOOL_NAME) missing in format node."
        print(f"❌ ERROR: {err_msg}")
        return {"error_message": err_msg, "is_finished": True}

    tool_results = {}
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):

            tool_results[msg.tool_call_id] = (
                str(msg.content) if msg.tool_call_id else config.PLACEHOLDER_IMAGE_URL
            )
        elif isinstance(msg, AIMessage) and msg.tool_calls:

            break

    final_recommendations_list = []

    for i, rec in enumerate(text_recommendations):
        if not isinstance(rec, dict):
            continue

        city = rec.get("city", "N/A")

        expected_call_id = (
            f"{tools.UNSPLASH_TOOL_NAME}_{i}_{city.replace(' ', '_').lower()}"
        )
        image_url = tool_results.get(expected_call_id, config.PLACEHOLDER_IMAGE_URL)

        if not isinstance(image_url, str) or not image_url.startswith("http"):
            final_image_url = config.PLACEHOLDER_IMAGE_URL
        else:
            final_image_url = image_url

        final_recommendations_list.append(
            {
                "city": city,
                "country": rec.get("country", ""),
                "description": rec.get("description", "No description available."),
                "justification": rec.get("justification", ""),
                "image_url": final_image_url,
                "has_data": rec.get("has_data", False),
            }
        )

    print(
        f"   Formatted {len(final_recommendations_list)} recommendations with image URLs."
    )

    return {
        "recommendations": final_recommendations_list,
        "messages": [
            AIMessage(content="Here are the recommendations I found for you:")
        ],
    }


def get_selection_node(state: schemas.SuggestionState) -> Dict[str, Any]:
    """
    Prompts the user to select a city OR automatically selects one in demo mode.
    Updates the state with the selection or signals exit.
    This node always transitions to END for the Phase 1 graph.
    """
    if state.get("is_finished"):
        return {}
    print("Requesting user selection...")

    recommendations = state.get("recommendations", [])
    if not recommendations:
        err_msg = "No final recommendations available to select from."
        print(f"❌ Error: {err_msg}")
        return {
            "is_finished": True,
            "error_message": err_msg,
            "selected_city_for_phase_2": None,
        }

    valid_choices_map = {
        rec["city"].lower(): rec["city"]
        for rec in recommendations
        if isinstance(rec, dict) and rec.get("has_data") and rec.get("city")
    }
    valid_cities_original_case = list(valid_choices_map.values())

    if not valid_choices_map:
        print("   None of the recommendations have detailed data available.")

        return {
            "is_finished": True,
            "error_message": "No recommendations with details available.",
            "selected_city_for_phase_2": None,
            "messages": [
                AIMessage(
                    content="Unfortunately, none of these options have detailed data available in my system right now. Let's end here."
                )
            ],
        }

    selected_city_original_case = None

    if config.INTERACTIVE_MODE:

        prompt_message = (
            "\nPlease choose a city with '✅ Details Available' to explore further:"
        )
        for city_name in valid_cities_original_case:
            prompt_message += f"\n - {city_name}"
        prompt_message += "\nOr type 'quit' to exit: "

        while True:
            try:
                user_input_raw = input(prompt_message)
                user_input_clean = user_input_raw.strip().lower()

                if user_input_clean == "quit":
                    print("Exiting.")
                    selected_city_original_case = None
                    break
                elif user_input_clean in valid_choices_map:

                    selected_city_original_case = valid_choices_map[user_input_clean]
                    print(
                        f"✅ Great! Preparing to explore {selected_city_original_case}..."
                    )
                    break
                else:
                    print(
                        f"'{user_input_raw}' is not a valid option with details. Please try again."
                    )

            except EOFError:
                print("\nInput stream closed, treating as quit.")
                selected_city_original_case = None
                break
            except Exception as e:
                print(f"❌ An error occurred during input: {e}. Exiting.")
                return {
                    "is_finished": True,
                    "error_message": f"Input error: {e}",
                    "selected_city_for_phase_2": None,
                }

    else:
        if valid_cities_original_case:

            selected_city_original_case = valid_cities_original_case[0]
            print(
                f"🤖 Demo Mode: Automatically selecting first valid city: {selected_city_original_case}"
            )
        else:

            print(
                "🤖 Demo Mode: No valid cities found to auto-select. Treating as quit."
            )
            selected_city_original_case = None

    return {
        "is_finished": True,
        "selected_city_for_phase_2": selected_city_original_case,
    }


def check_preferences_complete(
    state: schemas.SuggestionState,
) -> Union[Literal["ask_question", "generate_recommendations"], object]:
    """Routes based on preference collection status or if user quit."""
    if state.get("is_finished"):
        print("--> Route: END (User Quit/Error)")
        return END

    required_preferences = ["vibe", "activities", "weather", "budget"]
    collected_prefs = state.get("user_preferences", {}).keys()

    if all(pref in collected_prefs for pref in required_preferences):
        print("--> Route: generate_recommendations (Preferences Complete)")
        return "generate_recommendations"
    else:

        return "ask_question"


def route_after_tool_trigger(
    state: schemas.SuggestionState,
) -> Literal["image_tool_executor", "format_output"]:
    """Routes to tool executor if calls exist, otherwise skips to formatting."""

    last_message = state["messages"][-1] if state["messages"] else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("--> Route: image_tool_executor (Tool Calls Pending)")
        return "image_tool_executor"
    else:
        print("--> Route: format_output (No Tool Calls or Wrong Message Type)")

        return "format_output"


def build_suggestion_graph(unique_cities_lower_set: set):
    """Builds and compiles the LangGraph StateGraph for Phase 1."""

    if not initialize_phase1_llms():
        print("❌ Cannot build graph: Phase 1 LLM initialization failed.")
        sys.exit(1)

    graph_builder = StateGraph(schemas.SuggestionState)

    graph_builder.add_node("ask_question", ask_question_node)
    graph_builder.add_node("get_user_input", human_input_node)
    graph_builder.add_node("parse_preference", parse_preference_node)
    graph_builder.add_node("generate_recommendations", generate_recommendations_node)

    verifier_with_data = partial(
        verify_recommendations_node, unique_cities_lower=unique_cities_lower_set
    )
    graph_builder.add_node("verify_recommendations", verifier_with_data)

    graph_builder.add_node("call_image_tool", call_image_tool_node)

    graph_builder.add_node("image_tool_executor", ToolNode(tools.suggestion_tools))
    graph_builder.add_node("format_output", format_final_output_node)
    graph_builder.add_node("get_selection", get_selection_node)

    graph_builder.set_entry_point("ask_question")

    graph_builder.add_edge("ask_question", "get_user_input")
    graph_builder.add_edge("get_user_input", "parse_preference")

    graph_builder.add_conditional_edges(
        "parse_preference",
        check_preferences_complete,
        {
            "ask_question": "ask_question",
            "generate_recommendations": "generate_recommendations",
            END: END,
        },
    )

    graph_builder.add_edge("generate_recommendations", "verify_recommendations")
    graph_builder.add_edge("verify_recommendations", "call_image_tool")

    graph_builder.add_conditional_edges(
        "call_image_tool",
        route_after_tool_trigger,
        {
            "image_tool_executor": "image_tool_executor",
            "format_output": "format_output",
        },
    )

    graph_builder.add_edge("image_tool_executor", "format_output")

    graph_builder.add_edge("format_output", "get_selection")
    graph_builder.add_edge("get_selection", END)

    final_suggestion_graph = graph_builder.compile()

    print("✅ Phase 1 Suggestion agent graph compiled successfully.")
    return final_suggestion_graph
