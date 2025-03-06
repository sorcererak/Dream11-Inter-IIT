"""
The add_weather_to_csv function takes a CSV file containing cricket match data and appends weather data for each match.
It reads each row of match data, retrieves the relevant weather information using a helper function, and adds it to a 
new "weather" column. The updated data is then saved into a new CSV file.

Function Workflow:
1. Read the match data CSV file (assumed to have columns 'match_id', 'dates', and 'city').
2. Initialize a new "weather" column.
3. Iterate over each match row to:
    a. Extract 'match_id', 'dates', and 'city' (required for weather lookup).
    b. For each date in the match (dates may contain multiple days, e.g., for Test matches), call `get_weather_from_csv` 
       to fetch the average temperature and precipitation.
    c. Format the weather data as "avg_temp/precipitation" for each date and aggregate it as a comma-separated string.
    d. Append the aggregated weather data to the "weather" column in the row.
4. Write the updated rows with weather information to a new CSV file (prepends "updated_" to the original file name).
5. Print statements are included for debugging to verify that 'match_id' data is accessed correctly.

Example Usage:
add_weather_to_csv('final_matchdata_with_inning.csv')

Dependencies:
- The `get_weather_from_csv` function should be defined in the weather module. It accepts a CSV file with lat/long and 
  weather data - 'get_weather_from_csv('output_lat_lon.csv', city, date)' to retrieve average temperature and precipitation.
"""
import csv
import numpy as np
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
# Load the CSV into a numpy array (assuming no header and comma-delimited)
data = np.genfromtxt('output_lat_lon.csv', delimiter=',', dtype='str', skip_header=1)

# Initialize the Open-Meteo API client with cache and retry functionality
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_weather_data(latitude, longitude, date, variables=["temperature_2m", "relative_humidity_2m", "dew_point_2m"]):
    """
    Fetch hourly weather data for a specified location and date range.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        end_date (str): End date in 'YYYY-MM-DD' format.
        variables (list of str): List of requested weather variables (e.g., ["temperature_2m", "relative_humidity_2m"]).

    Returns:
        dict: Dictionary containing the requested weather variables for the specified date and time.
    """
    
    # Set up the request URL and parameters for Open-Meteo API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": variables
    }
    
    # Make the API request
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]  # Process the first location (extendable for multiple locations)

        # Process the hourly data
        hourly = response.Hourly()
        hourly_data = {
            "date": pd.to_datetime(hourly.Time(), unit="s", utc=True)
        }
        
        # Extract and store the requested weather variables
        weather_data = {}
        for i, variable in enumerate(variables):
            weather_data[variable] = hourly.Variables(i).ValuesAsNumpy()

        # Add weather data to the hourly_data dictionary
        hourly_data.update(weather_data)

        # Check the length of the data to ensure the index is valid
        data_length = len(hourly_data["temperature_2m"])
        

        # Return data at 3pm (index 15) if available, otherwise return available data
        if data_length > 15:
            temperature = float(hourly_data["temperature_2m"][15])  # 3pm data
            relative_humidity = float(hourly_data["relative_humidity_2m"][15])
            dew_point = float(hourly_data["dew_point_2m"][15])
        else:
            print("Data for 3pm (index 15) is not available. Returning last available data.")
            temperature = float(hourly_data["temperature_2m"][-1])  # Last available data
            relative_humidity = float(hourly_data["relative_humidity_2m"][-1])
            dew_point = float(hourly_data["dew_point_2m"][-1])

        return temperature, relative_humidity, dew_point

    except Exception as e:
        print(f"Error fetching data from Open-Meteo API: {e}")
        return None, None, None


# Define a function to get the 2nd and 3rd column values based on the 1st column
def get_lat_long_from_key(key):
    # Find the row where the first column matches the key
    row = data[data[:, 0] == key]
    if row.size > 0:
        lat, long = row[0, 1], row[0, 2]  # Get the 2nd and 3rd column values
        return lat, long
    else:
        return None, None

def add_weather_to_csv(csv_file):
    # Open the existing CSV file in read mode and a new file to write updated data
    with open(csv_file, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['weather']  # Add new weather column

        
        rows = []
        for row in reader:
            #match_id = row['match_id']  for progress review
            dates = row['dates'] 
            date_list = dates.split(", ") 
            city = row['city']  
            
            
            # Get the weather data for each date in the list
            weather_info = []
            for date in date_list:
                lat, long = get_lat_long_from_key(city)
                temp, rel_humidity, dew_point = get_weather_data(lat,long,date)
                weather_info.append(f"{temp}/{rel_humidity}/{dew_point}")

            # Join the weather information with commas and append it to the row
            row['weather'] = ', '.join(weather_info)  
            rows.append(row)

    # Write the updated rows back to a new CSV file
    with open("final_matchdata_with_weather.csv", mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

add_weather_to_csv('final_matchdata_with_inning.csv')
