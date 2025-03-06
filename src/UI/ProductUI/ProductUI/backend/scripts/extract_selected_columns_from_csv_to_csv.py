import pandas as pd

# Load the original CSV file
input_file = 'final_matchdata_with_weather.csv'  # Replace with the path to your input CSV file
output_file = 'weather.csv'  # Path to save the filtered CSV file

# List of columns to extract
columns_to_keep = ['match_id', 'weather']

# Read the CSV
df = pd.read_csv(input_file)

# Filter the desired columns
filtered_df = df[columns_to_keep]

filtered_df = filtered_df.drop_duplicates(subset='match_id') # Drop duplicate match_id rows
# Save the filtered columns to a new CSV
filtered_df.to_csv(output_file, index=False)

print(f"Filtered CSV saved as {output_file}")
