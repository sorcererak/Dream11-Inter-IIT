import pandas as pd
import ast
import re
import csv

def clean_team_name(team_name):
    """
    Clean individual team names by:
    - Removing unwanted patterns like 'Under-XXs', 'cricket', 'women', and ' A'.
    - Removing any parentheses or trailing characters like '()'.
    - Ensuring consistent formatting without extra spaces.
    """
    # Replace double quotes with single quotes
    team_name = team_name.replace('"', "'")
    
    # Remove unwanted patterns
    team_name = re.sub(r"Under-\d+s", "", team_name)  # Remove 'Under-XXs'
    team_name = team_name.replace("cricket", "").strip()  # Remove 'cricket'
    team_name = team_name.replace("Women", "").strip()  # Remove 'women' (case-insensitive)
    team_name = re.sub(r"\(\)", "", team_name).strip()  # Remove '()'
    
    # Remove ' A' if it's at the end or in any other part of the team name
    team_name = re.sub(r" A$", "", team_name)  # Remove ' A' at the end

    # Clean up extra spaces
    team_name = re.sub(r"\s{2,}", " ", team_name)

    return team_name

def clean_teams_list(teams):
    """
    Apply cleaning logic to each team name in the list.
    """
    return [clean_team_name(team) for team in teams]

def extract_player_teams_from_csv(csv_file_path, output_file_path):
    """
    Process the CSV to clean up team names and handle duplicates.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Parse the 'teams' column, which is stored as a string representation of a list
    df['teams'] = df['teams'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Clean up the team names
    df['teams'] = df['teams'].apply(clean_teams_list)

    # Group by player_id and aggregate the teams into a unique list for each player
    grouped_df = df.groupby('player_id')['teams'].apply(
        lambda x: list(set([team for sublist in x for team in sublist if team.strip()]))
    ).reset_index()

    # Convert the 'teams' column into a string representation of a list with single quotes
    grouped_df['teams'] = grouped_df['teams'].apply(lambda x: f"['{', '.join(x)}']")

    # Save the aggregated data to a new CSV file with proper formatting for lists
    grouped_df.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL)  # Adjust quoting without escape character

    print(f"Data successfully saved to {output_file_path}")

# Example usage
csv_file_path = "mw_pw_profiles.csv"  # Input CSV file
output_file_path = "teams_player_played_cleaned_final.csv"  # Output CSV file
extract_player_teams_from_csv(csv_file_path, output_file_path)
