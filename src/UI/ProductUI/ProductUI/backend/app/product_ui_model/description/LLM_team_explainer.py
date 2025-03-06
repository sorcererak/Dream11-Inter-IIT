import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pandas as pd
import numpy as np  # To fill nulls with np.nan

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)

import json
def Query_document(player_features, model="gemini-1.5-flash"):
    """
    Query Gemini API with player match data and generate a Python dictionary explaining team selection.
    
    Args:
        player_features (dict): A dictionary of player features.
        model (str): The Gemini model to query.

    Returns:
        dict: Python dictionary with explanations or an error message.
    """
    try:
        # Construct the prompt
        prompt = f"""
        You are a team_explainer for an app named Dream11 which predicts best fantasy team for today using ML algorithms trained on player stats and you are provided with the best 11 players for today's match predicted by us.
        Analyze the following player data to explain why each player was selected for the team. Include role-specific insights (batter, bowler, all-rounder) and justify the choice based on their performance metrics.
        Also since it is the best team predicted by us for today's match, so try to be as positive for the team as possible but be genuine (don't forge fake numbers) making it formal at the same time (we need best user experience) ,use some numbers (try to use some good name of feature) if you think they are good 
        enought and for the best 3 players try to emphasize more. Also try to describe each player differently so overall user feels satisfied by the team explaination.
        Generate a Python dictionary with the following structure:
        {{
            "team_explanation": "A positive and detailed explanation of the overall team strategy and balance.",
            "player_explanations": {{
                "player_id_1": "A detailed explanation about this player focusing on their role and recent performance.",
                "player_id_2": "Another explanation, emphasizing performance and consistency.",
                ...
            }}
        }}
        
        Here is the player data:
        """
        for player_id, features in player_features.items():
            prompt += f"""
            Player ID: {player_id}
            Role: {"Batter" if "avg_runs_scored" in features else "Bowler" if "wickets_taken" in features else "All-Rounder"}
            Key Features:
            """
            for feature, value in features.items():
                prompt += f"- {feature}: {value} ({get_feature_explanation(feature)})\n"

        prompt += f"""
        
        Make sure the output is a Python dictionary with valid syntax and keys for all 11 players plus the team explanation.
        """

        response = gemini_llm.predict(prompt)

        # Extract only the dictionary portion from the response
        import re
        match = re.search(r'(\{.*\})', response, re.DOTALL)  # Extract the dictionary from the response
        
        if match:
            response_dict = match.group(1)  # Get the matched dictionary portion
            result = eval(response_dict)    # Safely evaluate the dictionary string

            if isinstance(result, dict):
                return result
            else:
                raise ValueError("LLM did not return a valid dictionary.")
        else:
            raise ValueError("No valid dictionary found in the LLM response.")
            
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return {"error": "An error occurred while querying Gemini API."}




# Feature explanation function
def get_feature_explanation(feature):

    
    """Provide explanations for each feature."""
    explanations = {
    "avg_fantasy_score_15": "Average fantasy points scored over the last 15 matches, reflecting recent performance.",
    "centuries_cumsum": "Total number of centuries scored in the career, showcasing the player's ability to play big innings.",
    "order_seen_mode": "Most common batting position, indicating the player's familiarity with specific roles in the batting lineup.",
    "highest_runs": "The highest score achieved in an innings, representing the player's potential for game-changing performances.",
    "strike_rate_n3": "Scoring rate per 100 balls faced, calculated over the last three matches, highlighting the player's recent ability to score quickly.",
    "longter_avg_runs": "Average runs scored per match over the player's career, showing overall consistency in scoring.",
    "wickets_taken": "Total number of wickets taken across the player's career, indicating their impact as a bowler.",
    "economy_rate_n3": "Average runs conceded per over in the last three matches, reflecting the bowler's recent efficiency.",
    "bowling_average_n3": "Runs conceded per wicket taken, calculated over the last three matches, showing recent bowling effectiveness.",
    "longterm_avg_wickets_per_match": "Average number of wickets taken per match over the player's career, indicating overall consistency as a wicket-taker.",
    # "four_wicket_hauls_n": "Number of instances where the bowler has taken four or more wickets in a match, highlighting match-winning performances.",
    "CBR": "Career Bowling Rating, a comprehensive metric representing overall career performance as a bowler."
    }

    return explanations.get(feature, "Explanation not available.")

# Function to get player match details with features

def match_details_with_features(player_ids, odi):
    """Return match details with selected features for 11 players."""
    features = {
        "batter": ["full_name", "avg_fantasy_score_15", "centuries_cumsum", "order_seen_mode", "highest_runs", "strike_rate_n3", "longterm_avg_runs"],
        "bowler": ["full_name", "wickets_taken", "economy_rate_n3", "bowling_average_n3", "longterm_avg_wickets_per_match", "CBR"],
        "allrounder": [
            "full_name", "avg_fantasy_score_15", "centuries_cumsum", "order_seen_mode", 
            "wickets_taken", "economy_rate_n3", "bowling_average_n3"
        ]
    }
    
    player_features = {}
    for player_id in player_ids:
        player_match = get_last_row_by_player_name(odi, player_id)
        if player_match is not None:
            if player_match['batter'] == 1 and player_match['bowler'] == 0:
                selected_features = features["batter"]
            elif player_match['batter'] == 0 and player_match['bowler'] == 1:
                selected_features = features["bowler"]
            elif player_match['batter'] == 1 and player_match['bowler'] == 1:
                selected_features = features["allrounder"]
            else:
                # Default to allrounder features if role is indeterminate
                selected_features = features["allrounder"]

            # Extract the relevant features and store them in the dictionary
            player_features[player_id] = player_match[selected_features].to_dict()
        else:
            # If no data is found, default to allrounder and fill with nulls
            player_features[player_id] = {feature: None for feature in features["allrounder"]}
    
    return player_features

# Function to get the last row by player ID
def get_last_row_by_player_name(df, player_id):
    """Return the last row of the DataFrame where df['player_id'] equals the player ID."""
    filtered_df = df[df['player_id'] == player_id]
    if not filtered_df.empty:
        return filtered_df.iloc[-1]
    else:
        return None

# Main function to generate LLM responses
def add_performance_parameters(match_data, player_ids):
    """Generate performance explanations for the selected players."""
    player_features = match_details_with_features(player_ids, match_data)
    # print(player_features)
    if player_features:
        response = Query_document(player_features)
        return response
    else:
        return "No players found or invalid data."
    
# Main function to generate LLM responses
def add_performance_parameters(match_type, player_ids):
    """Generate performance explanations for the selected players."""
    
    # Map match types to their corresponding file names
    match_files = {
        "odi": "final_training_file_odi.csv",
        "test": "final_training_file_test.csv",
        "t20": "final_training_file_t20.csv",
        "mdm": "final_training_file_test.csv",
        "it20": "final_training_file_t20.csv",
        "odm": "final_training_file_odi.csv"
    }
    
    # Validate match type
    if match_type not in match_files:
        return "Invalid match type."
    
    # Construct file path
    current_dir = os.getcwd()+"/app/product_ui_model/src/data/procesed"
    print(current_dir)
    filename = match_files[match_type]
    file_path = os.path.join(current_dir, filename)
    print(file_path)
    
    try:
        # Load match data
        match_data = pd.read_csv(file_path)
    except FileNotFoundError:
        return f"File {filename} not found in {current_dir}."
    except pd.errors.EmptyDataError:
        return f"File {filename} is empty."
    except Exception as e:
        return f"An error occurred while reading {filename}: {e}"
    
    # Generate player features
    player_features = match_details_with_features(player_ids, match_data)
    
    # Return response based on player features
    if player_features:
        response = Query_document(player_features)
        return response
    else:
        return "No players found or invalid data."
    

# # Example usage
# current_dir = os.getcwd()
# filename = "odi_final[1].csv"
# file_path = os.path.join(current_dir, filename)
# match_data = pd.read_csv(file_path)
# match_data = pd.read_csv("/home/harshit/Desktop/Dream_11_chatbot/odi_final[1].csv")

# player_ids = [
#     '43936951', '7fb32e5b', 'b8a55852', '3eac9d95', '1b668884',
#     '8b5b6769', '99639abf', '91a4a398', '2764133a', '8ba8195d', 'bad31fac'
# ]

# performance_responses = add_performance_parameters(match_data, player_ids)
# print(performance_responses)
