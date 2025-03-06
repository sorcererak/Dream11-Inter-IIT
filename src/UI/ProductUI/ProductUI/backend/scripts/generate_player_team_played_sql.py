import csv

def csv_to_sql_dump(csv_file_path, sql_dump_file_path):
    # Define the table schema
    create_table_query = """
    -- SQL Dump File for PostgreSQL
    CREATE TABLE players (
        player_id VARCHAR(255) NOT NULL,
        teams TEXT[] NOT NULL
    );
    """

    # Open the CSV file and prepare the SQL dump
    with open(csv_file_path, 'r') as csv_file, open(sql_dump_file_path, 'w') as sql_file:
        csv_reader = csv.reader(csv_file)

        # Write the CREATE TABLE statement
        sql_file.write(create_table_query + "\n")

        # Start the INSERT statement
        insert_query = "INSERT INTO players (player_id, teams) VALUES\n"
        sql_file.write(insert_query)

        # Process rows from the CSV file
        rows = []
        for row in csv_reader:
            player_id = row[0]
            teams = row[1].strip("[]").replace("\"", "").replace("'", "").split(", ")
            teams_array = "ARRAY['" + "', '".join(teams) + "']"
            rows.append(f"('{player_id}', {teams_array})")

        # Write all rows, joined by commas, and end with a semicolon
        sql_file.write(",\n".join(rows) + ";\n")

# Specify the input CSV file and output SQL file
csv_file_path = 'teams_player_played_cleaned_final.csv'  # Replace with the path to your CSV file
sql_dump_file_path = 'players_dump_played.sql'  # Replace with the desired output SQL file path

# Convert the CSV to SQL dump
csv_to_sql_dump(csv_file_path, sql_dump_file_path)
