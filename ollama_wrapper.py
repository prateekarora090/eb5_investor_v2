import requests
from typing import Any, List, Mapping, Optional, Dict
from pydantic import BaseModel, Field
from langchain.llms.base import LLM

class OllamaConfig(BaseModel):
    model_name: str
    api_url: str = Field(default="http://localhost:11434/api/generate")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    top_p: float = Field(default=0.95)

class OllamaWrapper(LLM):
    config: OllamaConfig

    def __init__(self, **kwargs):
        config = OllamaConfig(**kwargs)
        super().__init__(config=config)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generates text using the Ollama API."""
        data = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }
        if stop:
            data["stop"] = stop

        response = requests.post(self.config.api_url, json=data)
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            raise Exception(f"Error from Ollama API: {response.text}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.config.model_name}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.config.model_name}