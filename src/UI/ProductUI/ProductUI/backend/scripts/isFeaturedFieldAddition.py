import pandas as pd

# Load the CSV file into a pandas DataFrame
file_path = "path_to_your_file/unique_event.csv"
df = pd.read_csv(file_path)

# Show the contents of the CSV
print(df.head())  # This will print the first few rows of your file for confirmation

# Create a dictionary mapping event_name to isFeatured
event_is_featured = {}

# Iterate over the dataframe to create the mapping
for index, row in df.iterrows():
    event_name = row['event_name']
    is_featured = row['isFeatured'] if pd.notna(row['isFeatured']) else 'no'  # Default to 'no' if empty
    event_is_featured[event_name] = is_featured

# Check the created dictionary
print(event_is_featured)

from sqlalchemy.orm import Session

def update_is_featured_from_file(db: Session, event_is_featured: dict):
    # Iterate through each event name and isFeatured value
    for event_name, is_featured in event_is_featured.items():
        # Update the matches where the event_name matches the current event
        db.query(model.Match).filter(model.Match.event_name == event_name) \
            .update({model.Match.isFeatured: is_featured}, synchronize_session='fetch')

    # Commit changes to the database
    db.commit()

# Example usage of the function
# Assuming you have a valid `db` session
# update_is_featured_from_file(db, event_is_featured)

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session


