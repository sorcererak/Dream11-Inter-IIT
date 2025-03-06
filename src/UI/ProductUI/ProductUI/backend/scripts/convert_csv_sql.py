import pandas as pd
import sqlite3

# Load data from CSV file
csv_file = "player_data.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Add an 'id' column as primary key
# data.insert(0, 'id', range(1, len(data) + 1))  # Create a primary key column 'id' (starting from 1)

# Connect to an SQLite database (or create one)
db_file = "output1.db"
conn = sqlite3.connect(db_file)

# Create the table with 'id' as the primary key
table_name = "player_lifetime_stats"  # Name of the table in SQL

# Using `if_exists="replace"` to create the table if not already present, 
# and adding the 'id' column as the primary key
data.to_sql(table_name, conn, if_exists="replace", index=False)

# Export the database to an SQL dump
dump_file = "player_lifetime_stats.sql"
with open(dump_file, "w") as f:
    for line in conn.iterdump():
        f.write(f"{line}\n")

# Close the connection
conn.close()

print(f"SQL dump created at: {dump_file}")
