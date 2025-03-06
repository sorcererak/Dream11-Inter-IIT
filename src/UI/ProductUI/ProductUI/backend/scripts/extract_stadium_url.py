import requests
from bs4 import BeautifulSoup

def get_stadium_url(stadium_name):
    # Create the search URL based on the stadium name
    search_url = f"https://www.espncricinfo.com/cricket-grounds/{stadium_name.replace(' ', '-').lower()}"

    # Send request to get the page content
    response = requests.get(search_url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error fetching page for {stadium_name}. Status code: {response.status_code}")
        return None

def parse_stadium_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find relevant data on the page (you need to adjust this depending on the page's structure)
    data = {}
    
    # Example of scraping the stadium name (or other info)
    stadium_name_tag = soup.find('h1', class_='headline')
    if stadium_name_tag:
        data['stadium_name'] = stadium_name_tag.text.strip()
    
    # You can find other relevant details such as capacity, city, etc.
    # Example: scraping the city and country
    city_country_tag = soup.find('div', class_='stadium-header__location')
    if city_country_tag:
        data['city_country'] = city_country_tag.text.strip()

    # You can add more extraction points based on the website structure
    return data

def main(stadium_name):
    html_content = get_stadium_url(stadium_name)
    
    if html_content:
        stadium_data = parse_stadium_data(html_content)
        print(stadium_data)

if __name__ == "__main__":
    stadium_name = input("Enter the stadium name: ")
    main(stadium_name)
