import csv
from player_details_with_images import player_details_with_images
# Data to be converted


# Filepath for output CSV
output_csv_path = 'csv/player_details.csv'

# Writing data to CSV
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=player_details_with_images[0].keys())
    writer.writeheader()
    writer.writerows(player_details_with_images)

output_csv_path