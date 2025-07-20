# autovisory_app.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Autovisory AI Demo",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

# ==============================================================================
# API AND MODEL CONFIGURATION
# ==============================================================================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    st.error("Could not configure Google AI. Have you set the GOOGLE_API_KEY in Streamlit secrets?", icon="🚨")
    st.stop()

# ==============================================================================
# DATA LOADING (with Caching for performance)
# ==============================================================================
@st.cache_data
def load_data():
    try:
        df_new_us = pd.read_csv('Data/sample_new_us_cars.csv')
        df_used_us = pd.read_csv('Data/sample_used_us_cars.csv')
        df_used_europe = pd.read_csv('Data/sample_used_europe_cars.csv')
        return df_new_us, df_used_us, df_used_europe
    except FileNotFoundError:
        st.error("Sample data files not found. Please ensure the 'Data' folder with your sample CSVs is in the repository.", icon="🚨")
        return None, None, None

df_new_us_master, df_used_us_master, df_used_europe_master = load_data()
if df_new_us_master is None:
    st.stop()


# ==============================================================================
# ----------------- AI HELPER FUNCTIONS (FINAL VERSION) ----------------------
# ==============================================================================

def determine_next_action(history, user_query):
    history_str = "\n".join([f"{h['role']}: {h['parts']}" for h in history])
    
    # FINAL, POLISHED PROMPT WITH SMALL TALK HANDLING
    prompt = f"""
    You are Autovisory, a helpful and conversational AI car advisor. Your goal is to classify the user's most recent query into a specific action.

    Based on the user's most recent query, determine the primary intent by following these rules IN ORDER:

    1.  **Small Talk:** If the query is a simple greeting ('hi', 'hello'), farewell ('bye'), expression of gratitude ('thanks', 'thank you'), or other common pleasantries that DO NOT contain a car request, your action is "small_talk". Generate an appropriate, polite response.
        - Example for "thanks": {{"action": "small_talk", "response": "You're very welcome! Is there anything else I can help you analyze or compare?"}}
        - Example for "hello": {{"action": "small_talk", "response": "Hello! How can I help you with your car search today?"}}

    2.  **Clarify:** If the query is a vague car-related request (e.g., "I need a car"), your action is "clarify".
        - JSON: {{"action": "clarify", "response": "To give you the best recommendation, I need a little more information. Could you tell me about your budget, primary use, and priorities?"}}

    3.  **Recommend:** If the user provides details for a recommendation OR asks to modify previous criteria (e.g., "what about something cheaper?", "change my preference to sport cars"), your action is "recommend".
        - JSON: {{"action": "recommend", "response": "Okay, based on your new preferences, I'm finding some options for you..."}}

    4.  **Analyze:** If the user asks for details about ONE specific car model, your action is "analyze".
        - JSON: {{"action": "analyze", "response": "Let me pull up the details for that model."}}

    5.  **Compare:** If the user asks to compare TWO OR MORE specific car models, your action is "compare".
        - JSON: {{"action": "compare", "response": "Excellent comparison. Let me put the specs side-by-side."}}
        
    6.  **Answer General:** If the user asks a general knowledge question about cars (e.g., "What is a hybrid?"), your action is "answer_general".
        - JSON: {{"action": "answer_general", "response": "That's a great question. Here's an explanation..."}}

    7.  **Reject:** If the query is CLEARLY not about cars (e.g., "What's the weather?"), your action is "reject".
        - JSON: {{"action": "reject", "response": "I'm designed to only answer questions about cars. Could we stick to that topic?"}}

    Conversation History:
    {history_str}

    User's Latest Query: "{user_query}"

    Return only the single, valid JSON object for your chosen action.
    """
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
            return json.loads(text)
        except Exception:
            continue
    return {"action": "error", "response": "Sorry, I had trouble understanding that. Could you please rephrase?"}


def extract_car_models(text):
    pattern = r'(?:\b(?:vs|versus|compare|between|and)\b\s)?([A-Z][a-zA-Z0-9-]+\s(?:[A-Z][a-zA-Z0-9-]+-?)+|[A-Z][a-zA-Z0-9-]+\s[A-Z][a-zA-Z0-9-]+|[A-Z][a-zA-Z0-9-]+)'
    models = re.findall(pattern, text)
    stop_words = {'Compare', 'Between', 'And', 'The', 'A'}
    return [model.strip() for model in models if model.strip() not in stop_words]


def get_recommendations_and_analysis(full_context_query):
    prompt = f"""
    You're an expert AI Car Analyst. Based on the user's request, recommend 3 cars and provide an analysis.

    FULL CONVERSATION CONTEXT:
    {full_context_query}

    INSTRUCTIONS:
    1.  Analyze the user's needs from the context.
    2.  Select 3 car models from your knowledge that are the best fit.
    3.  For each car, provide a compelling summary and an estimated price range.
    4.  You MUST respond in a valid JSON object like this example. Price must be an integer.

    EXAMPLE JSON RESPONSE:
    {{
      "recommendations": [
        {{
          "make": "Toyota",
          "model": "Camry",
          "summary": "The Toyota Camry is a fantastic choice for its legendary reliability, excellent fuel economy, and strong safety scores. It's a comfortable and practical midsize sedan that holds its value well.",
          "price_range": {{
            "min_price": 25000,
            "max_price": 35000,
            "type": "New"
          }}
        }},
        {{
          "make": "Ford",
          "model": "F-150",
          "summary": "If the user needs a used truck, the Ford F-150 is the market leader. It's known for its wide range of configurations, strong towing capacity, and a comfortable ride.",
          "price_range": {{
            "min_price": 25000,
            "max_price": 40000,
            "type": "Used (3-5 years old)"
          }}
        }}
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "recommendations": []}


def compare_cars_with_ai(full_context_query):
    prompt = f"""
    You are a car expert AI. The user is trying to decide between two or more vehicles.
    Based on this conversation, create a side-by-side comparison.

    FULL CONVERSATION CONTEXT:
    {full_context_query}

    INSTRUCTIONS:
    1. Identify the car models the user wants to compare.
    2. Provide a brief summary for each model.
    3. List 2-3 key strengths and 2-3 key weaknesses for each.
    4. Respond ONLY with a valid JSON object like the example below.

    EXAMPLE JSON RESPONSE:
    {{
      "comparison": [
        {{
          "model": "Honda Civic",
          "summary": "A compact car known for its sporty handling, fuel efficiency, and high reliability ratings. It's a great all-rounder for singles or small families.",
          "strengths": ["Fun-to-drive dynamics", "Excellent fuel economy", "High resale value"],
          "weaknesses": ["Road noise can be high at speed", "Base model is light on features"]
        }},
        {{
          "model": "Toyota Corolla",
          "summary": "The Corolla's reputation is built on reliability, comfort, and safety. It prioritizes a smooth ride and ease of use over sporty performance.",
          "strengths": ["Legendary reliability", "Standard safety features", "Comfortable ride"],
          "weaknesses": ["Uninspired engine performance", "Less engaging to drive than rivals"]
        }}
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "comparison": []}


def analyze_specific_car_model(car_model):
    prompt = f"""
    You are an expert automotive analyst. Give a clear, concise analysis of the following car model.

    MODEL TO ANALYZE: "{car_model}"

    INSTRUCTIONS:
    1.  Provide a one-paragraph overview of what the car is known for.
    2.  List 3 distinct pros and 3 distinct cons.
    3.  Describe the target audience for this vehicle.
    4.  Provide a typical market price range.
    5.  Respond ONLY in the following valid JSON format.

    EXAMPLE JSON RESPONSE:
    {{
      "model": "Tesla Model Y",
      "overview": "The Tesla Model Y is a fully electric compact SUV that has become incredibly popular for its blend of long-range capability, cutting-edge technology, and impressive performance. It shares many components with the Model 3 sedan but offers more practicality with its hatchback design and available third-row seat.",
      "pros": ["Impressive real-world battery range", "Access to Tesla's reliable Supercharger network", "Quick acceleration and nimble handling"],
      "cons": ["Stiff ride quality, especially on larger wheels", "Reliance on the touchscreen for most controls can be distracting", "Build quality can be inconsistent compared to legacy automakers"],
      "audience": "Tech-savvy individuals and families looking for a practical EV with a focus on performance and access to the best charging infrastructure.",
      "price_estimate_usd": "$45,000 - $60,000"
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# STREAMLIT CHAT UI
# ==============================================================================
st.title("🚗 Autovisory AI")
st.caption("Your AI-Powered Car Market Analyst (Live Demo)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Autovisory. Ask me to recommend, compare, or analyze a car."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            gemini_history = [{"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]} for msg in st.session_state.messages]
            
            action_data = determine_next_action(gemini_history, prompt)
            action = action_data.get("action", "error")
            response_content = ""

            # UPDATED LOGIC BLOCK TO HANDLE THE NEW 'small_talk' ACTION
            if action in ["reject", "clarify", "answer_general", "small_talk"]:
                response_content = action_data.get("response", "I'm not sure how to respond. Please try rephrasing.")
            
            elif action == "recommend":
                full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                recs = get_recommendations_and_analysis(full_context)
                if recs.get("recommendations"):
                    response_content = "Based on your preferences, here are 3 solid options:\n"
                    for r in recs["recommendations"]:
                        response_content += f"\n### 🚗 {r.get('make')} {r.get('model')}\n"
                        response_content += f"- **Summary**: {r.get('summary', 'N/A')}\n"
                        
                        price_info = r.get('price_range', {})
                        min_p = price_info.get('min_price', 0)
                        max_p = price_info.get('max_price', 0)
                        type_p = price_info.get('type', 'N/A')
                        
                        if min_p > 0 and max_p > 0:
                            response_content += f"- **Estimated Price**: ${min_p:,} - ${max_p:,} ({type_p})\n"
                        else:
                            response_content += "- **Estimated Price**: Not available\n"
                else:
                    response_content = "Sorry, I couldn't find good options with the provided details. Could you be more specific?"

            elif action == "analyze":
                candidates = extract_car_models(prompt)
                model_name = candidates[0] if candidates else ""
                if model_name:
                    analysis = analyze_specific_car_model(model_name)
                    if analysis.get("model"):
                        response_content = f"### Analysis of the {analysis['model']}\n"
                        response_content += f"**📘 Overview:** {analysis['overview']}\n\n"
                        response_content += "**✅ Pros:**\n"
                        for pro in analysis.get('pros', []):
                            response_content += f"- {pro}\n"
                        response_content += "\n**⚠️ Cons:**\n"
                        for con in analysis.get('cons', []):
                            response_content += f"- {con}\n"
                        response_content += f"\n**👥 Ideal For:** {analysis.get('audience', 'N/A')}\n"
                        response_content += f"**💰 Estimated Price:** {analysis.get('price_estimate_usd', 'N/A')}"
                    else:
                        response_content = "Sorry, I couldn't analyze that model. Please be more specific."
                else:
                    response_content = "I couldn't identify a specific car model to analyze. Please try again."

            elif action == "compare":
                full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                result = compare_cars_with_ai(full_context)
                if result.get("comparison"):
                    response_content = "Here's a comparison of your choices:\n"
                    for car in result["comparison"]:
                        response_content += f"\n### 🚘 {car['model']}\n"
                        response_content += f"- **Summary**: {car.get('summary', 'N/A')}\n"
                        response_content += f"- **✅ Strengths**: {', '.join(car.get('strengths', []))}\n"
                        response_content += f"- **⚠️ Weaknesses**: {', '.join(car.get('weaknesses', []))}\n"
                else:
                    response_content = "Sorry, I couldn't generate a comparison. Please mention at least two models clearly."
            
            else:
                response_content = action_data.get("response", "I encountered an issue. Please try again.")

            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
