from crewai_tools import BaseTool
# NOTE: Alternative implementation could be using "from langchain.tools import tool"

import os
import requests
import json
from typing import Type, Any
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field

# Load environment variables
load_dotenv('secrets/.env') ## for os.getenv('SERPER_API_KEY') below

class WebSearchToolSchema(BaseModel):
    """Input for WebSearchTool."""
    search_query: str = Field(..., description="Mandatory search query to search the internet.")
    n_results: str = Field(..., description="Optional param to customize the number of desired results.")

class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the internet for up-to-date information with an input search_query."
    args_schema: Type[BaseModel] = WebSearchToolSchema
    n_results: int = 6

    def _run(self, **kwargs: Any) -> Any:
        query = kwargs.get('search_query')
        if query is None:
            query = kwargs.get('query') # fallback, in case wrongfully specified
            if query is None:
                print("[WebSearchTool] ERROR: No query specified for searching!")
                return

        n = kwargs.get('n_results')

        # TODO: Actually use "n" to return the number of results the agent asked for.
        print(f"Parsed n to be {n}.")
        
        api_key = os.getenv('SERPER_API_KEY')
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        
        if response.status_code == 200:
            return self._parse_results(response.json())
        else:
            return f"Error: Unable to fetch results (Status code: {response.status_code})"

    def _parse_results(self, json_response):
        organic_results = json_response.get('organic', [])
        parsed_results = []
        
        for result in organic_results[:3]:  # Limiting to top 3 results
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No snippet available')
            link = result.get('link', 'No link available')
            parsed_results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")
        
        return "\n".join(parsed_results)