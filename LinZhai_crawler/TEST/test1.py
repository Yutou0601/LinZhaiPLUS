import requests
from bs4 import BeautifulSoup
response = requests.get("http://www.yahoo.com")
soup = BeautifulSoup(response.text, "html.parser")

print(soup.title)
