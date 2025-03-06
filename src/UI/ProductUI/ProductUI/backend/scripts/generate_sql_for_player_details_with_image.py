from player_details_with_images import player_details_with_images


# # Define the SQL file path
# sql_file_path = "player_details_dump.sql"

# # Create table SQL statement
# create_table_query = """
# CREATE TABLE players (
#     player_id TEXT PRIMARY KEY,
#     unique_name TEXT,
#     key_cricinfo REAL,
#     full_name TEXT,
#     gender TEXT,
#     playing_role TEXT,
#     bg_image_url TEXT,
#     img_src_url TEXT
# );
# """

# # Generate insert statements
# insert_queries = []
# for player in player_details_with_images:
#     query = f"""
#     INSERT INTO players (
#         player_id, unique_name, key_cricinfo, full_name, gender, playing_role, bg_image_url, img_src_url
#     ) VALUES (
#         '{player['player_id']}', '{player['unique_name']}', {player['key_cricinfo']}, 
#         '{player['full_name']}', '{player['gender']}', '{player['playing_role'] or 'NULL'}', 
#         '{player['bg_image_url'] or 'NULL'}', '{player['img_src_url'] or 'NULL'}'
#     );
#     """
#     insert_queries.append(query.strip())

# # Combine all SQL commands
# sql_script = create_table_query + "\n" + "\n".join(insert_queries)

# # Write to the SQL file
# with open(sql_file_path, "w") as sql_file:
#     sql_file.write(sql_script)

# print(f"SQL dump file created at: {sql_file_path}")


# SQL Dump File Name
output_file = "players_profile_dump.sql"

# Table Name
table_name = "players"

# Create SQL Dump
with open(output_file, "w") as file:
    # Write SQL to create the table
    file.write(f"""CREATE TABLE {table_name} (
        player_id TEXT PRIMARY KEY,
        unique_name TEXT,
        key_cricinfo REAL,
        full_name TEXT,
        gender TEXT,
        playing_role TEXT,
        bg_image_url TEXT,
        img_src_url TEXT
    );\n\n""")

    # Insert data
    for player in player_details_with_images:
        sql = f"""INSERT INTO {table_name} (player_id, unique_name, key_cricinfo, full_name, gender, playing_role, bg_image_url, img_src_url)
        VALUES (
            '{player['player_id']}',
            '{player['unique_name'].replace("'", "''")}',
            '{player['key_cricinfo']}',
            '{player['full_name'].replace("'", "''")}',
            '{player['gender']}',
            '{player['playing_role']}',
            '{player['bg_image_url']}',
            '{player['img_src_url']}'
        );\n"""
        file.write(sql)

print(f"SQL dump successfully written to {output_file}")