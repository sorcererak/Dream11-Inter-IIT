import pandas as pd

# Load the CSV file
input_file = 'weather.csv'  # Replace with the path to your input CSV file
output_file = 'weather.sql'  # Path to save the SQL dump file

# Read the CSV file
df = pd.read_csv(input_file)

# Function to convert weather data to nested arrays
def convert_weather_to_array(weather_str):
    # Handle case where weather_str is 'None'
    if weather_str == 'None' or weather_str is None:
        return []

    # Split by commas to get individual sets
    sets = weather_str.split(', ')
    
    # Convert each set into an array by splitting by '/' and handling possible conversion errors
    nested_array = []
    for s in sets:
        try:
            # Attempt to convert each part to a float, replace 'None' with 'null' in JSON format
            nested_array.append([None if x == 'None' else float(x) for x in s.split('/')])
        except ValueError:
            # If a value can't be converted to float, we append [null, null, null]
            nested_array.append([None, None, None])
    
    # Replace None with null in the nested array to make it valid JSON
    return str(nested_array).replace('None', 'null')

# Open the output file to write SQL dump
with open(output_file, 'w') as sql_file:
    # Write CREATE TABLE command
    sql_file.write("""
    CREATE TABLE IF NOT EXISTS weather_data (
        match_id BIGINT PRIMARY KEY,
        weather JSONB
    );
    \n""")

    # Iterate over each row in the dataframe to generate the INSERT statements
    for index, row in df.iterrows():
        match_id = row['match_id']
        weather_str = row['weather']
        # Convert weather data to nested array format
        nested_array = convert_weather_to_array(weather_str)
        
        # Prepare the SQL insert statement in a single line with ON CONFLICT to handle duplicate keys
        sql_statement = f"INSERT INTO weather_data (match_id, weather) VALUES ({match_id}, '{nested_array}') ON CONFLICT (match_id) DO NOTHING;\n"
        
        # Write the SQL insert statement to the file
        sql_file.write(sql_statement)

print(f"SQL dump has been saved as {output_file}")
