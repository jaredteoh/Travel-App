import os
import requests

import streamlit as st
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


# GOOGLE_PLACES_API_KEY = st.secrets["GOOGLE_PLACES_API_KEY"]


# def get_location_suggestions(query):
#     url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json"
#     params = {
#         "input": query,
#         "types": "geocode",
#         "key": GOOGLE_PLACES_API_KEY
#     }
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         predictions = response.json().get("predictions", [])
#         return [prediction["description"] for prediction in predictions]
#     return []


def get_external_travel_data(destination):
    wikimedia_api_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": destination,
        "exintro": True,
        "explaintext": True,
    }

    response = requests.get(wikimedia_api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id != "-1":  # Check if valid page found
                return page_data.get("extract", "No additional information available.")
    return "No external data available for this destination."


def main():
    llm = Ollama(model="llama3")

    prompt_template = """
    You are an AI travel planner. The user is traveling to {destination} and has the following preferences:
    
    {preferences}

    The user will arrive on {arrival_date} at {arrival_time} and depart on {departure_date} at {departure_time}.
    
    Create a detailed itinerary that balances the user's preferences. Ensure the following:
    - Activities start after the user's arrival time and finish well before departure time.
    - If the user arrives late (e.g., after 9 PM), only suggest light activities such as meals or evening walks for that day.
    - If the user departs early (e.g., before 9 AM), avoid scheduling activities on those days, except for meals or checking out.
    - Include specific places to visit, recommended activities, dining options, and suitable accommodations.
    - Allow reasonable time gaps for travel, meals, and rest between activities.
    - Use the following external data about {destination}:
        {external_data}
    """
    
    prompt = PromptTemplate(
        input_variables=[
            "destination",
            "preferences",
            "arrival_date",
            "arrival_time",
            "departure_date",
            "departure_time",
        ],
        template=prompt_template
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    st.title("AI Travel Planner")

    # input for travel destination
    location_query = st.text_input("Enter your travel destination:")

    # input for arrival and departure info
    arrival_date = st.date_input("Arrival Date")
    departure_date = st.date_input("Departure Date")
    arrival_time = st.time_input("Arrival Time")
    departure_time = st.time_input("Departure Time")

    if arrival_date and departure_date:
        trip_duration = (departure_date - arrival_date).days + 1
    else:
        trip_duration = 0

    activity_categories = {
        "Outdoor 🌲": ["Hiking", "Camping", "Beach", "Water Sports"],
        "Shopping 🛍️": ["Malls", "Antique Shops", "Luxury Brands", "Street Markets"],
        "Food & Drink 🍽️": ["Local Cuisine", "Fine Dining", "Street Food", "Wineries", "Breweries"],
        "Cultural 🎨": ["Museums", "Historical Sites", "Festivals", "Art Galleries"],
        "Adventure 🧗": ["Skydiving", "Scuba Diving", "Mountain Climbing"]
    }

    st.subheader("What type of activities do you enjoy?")
    selected_categories = st.multiselect(
        "Select general activities you enjoy:", list(activity_categories.keys())
    )

    selected_sub_options = {}
    for category in selected_categories:
        selected_sub_options[category] = st.multiselect(
            f"Select your preferences for {category.lower()}:",
            options=activity_categories[category]
        )

    if st.button("Generate Travel Plan"):
        if location_query and trip_duration > 0:
            preferences = []
            for activity, sub_activities in selected_sub_options.items():
                if sub_activities:
                    preferences.append(f"{activity}: {', '.join(sub_activities)}")
            
            # only include preferences if there are any
            if preferences:
                preferences_str = "; ".join(preferences)
            else:
                preferences_str = "No specific preferences provided."

            # RAG: retrieve external travel data from Wikimedia
            external_data = get_external_travel_data(location_query)

            travel_plan = llm_chain.run(
                {
                    "destination": location_query,
                    "preferences": preferences_str,
                    "arrival_date": arrival_date.strftime("%Y-%m-%d"),
                    "arrival_time": str(arrival_time),
                    "departure_date": departure_date.strftime("%Y-%m-%d"),
                    "departure_time": str(departure_time),
                    "external_data": external_data,
                }
            )
            st.subheader("Your AI-Generated Travel Plan:")
            st.write(travel_plan)
        else:
            st.error("Please enter a destination, and ensure the trip duration is valid!")


if __name__ == "__main__":
    main()
