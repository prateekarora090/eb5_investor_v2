from crewai_tools import BaseTool
from typing import Type, Any
from pydantic.v1 import BaseModel, Field
import requests
from bs4 import BeautifulSoup

class WebScraperToolSchema(BaseModel):
    """Input for WebScraperTool."""
    url: str = Field(..., description="URL of the website to scrape.")

class WebScraperTool(BaseTool):
    name: str = "Web Scraper"
    description: str = "Scrapes text content from a given website URL."
    args_schema: Type[BaseModel] = WebScraperToolSchema

    def _run(self, **kwargs: Any) -> Any:
        url = kwargs.get("url")
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text from all paragraphs
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            return text
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"