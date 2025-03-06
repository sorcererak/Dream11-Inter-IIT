import csv
import requests

# API Endpoint
API_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Input and Output Files
input_csv = "final_matchdata_with_inning.csv"  # Input CSV with city names
output_csv = "output_lat_lon_1.csv"  # Output CSV with city, latitude, longitude

# Function to read city names from a CSV file
import csv

def read_cities(file_path):
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)  # Use DictReader for header-based access
        cities = {row['city'] for row in reader if 'city' in row and row['city']}  # Use a set for unique values
    return list(cities)  # Convert back to a list if needed

# Function to fetch latitude and longitude from Open-Meteo API
def fetch_coordinates(city_name):
    try:
        response = requests.get(API_URL, params={"name": city_name, "count": 1, "language": "en", "format": "json"})
        response.raise_for_status()  # Raise an error for bad HTTP responses
        data = response.json()
        if "results" in data and data["results"]:
            result = data["results"][0]
            return result["latitude"], result["longitude"]
        else:
            print(f"No coordinates found for: {city_name}")
            return None, None
    except requests.RequestException as e:
        print(f"Error fetching data for {city_name}: {e}")
        return None, None

# Function to write geocoded data to a CSV file
def write_to_csv(file_path, data):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Main Function
if __name__ == "__main__":
    cities = read_cities(input_csv)
    geocoded_data = []

    for city in cities:
        latitude, longitude = fetch_coordinates(city)
        geocoded_data.append([city, latitude, longitude])
        #print(f"Processed: {city}, Latitude: {latitude}, Longitude: {longitude}")

    write_to_csv(output_csv, geocoded_data)
    #print(f"Geocoded data saved to {output_csv}")
