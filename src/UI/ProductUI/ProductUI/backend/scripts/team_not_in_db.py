import pandas as pd
import ast
from unique_teams import unique_teams

def filter_unlisted_teams(cleaned_csv_path, unique_teams, output_file_path):
    """
    Compare teams from the cleaned CSV file with the unique_teams list
    and generate a new CSV containing players whose teams are not in unique_teams.
    """
    # Load the cleaned player-teams CSV
    df = pd.read_csv(cleaned_csv_path)

    # Convert the teams column to lists safely
    def parse_teams(teams_str):
        try:
            return ast.literal_eval(teams_str) if isinstance(teams_str, str) else []
        except (ValueError, SyntaxError):
            print(f"Error parsing teams: {teams_str}")
            return []

    df['teams'] = df['teams'].apply(parse_teams)

    # Define the list of unique teams
    unique_teams_set = set(unique_teams)

    # Filter out teams not in the unique_teams list
    df['unlisted_teams'] = df['teams'].apply(
        lambda teams: [team for team in teams if team not in unique_teams_set]
    )

    # Retain only rows where there are unlisted teams
    filtered_df = df[df['unlisted_teams'].apply(lambda x: len(x) > 0)]

    # Save the filtered data to a new CSV
    filtered_df.to_csv(output_file_path, index=False)

    print(f"Data successfully saved to {output_file_path}")

# Example usage
cleaned_csv_path = "teams_player_played_cleaned_final.csv"  # Input cleaned CSV
output_file_path = "players_with_unlisted_teams.csv"  # Output CSV for unlisted teams
filter_unlisted_teams(cleaned_csv_path, unique_teams, output_file_path)
