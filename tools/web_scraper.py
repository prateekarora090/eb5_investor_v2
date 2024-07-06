import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs as an example
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"