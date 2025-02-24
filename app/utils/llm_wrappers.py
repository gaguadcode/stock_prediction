from app.utils.llm import BaseLLM
from typing import Type
import sys
import ollama
import google.generativeai as genai
from app.utils.config import config
'''
IMPORTS = {}
# Try importing each module individually and print specific errors if they fail
for module_name in ["openai", "anthropic", "cohere", "ollama"]:
    try:
        IMPORTS[module_name] = __import__(module_name)
    except ImportError:
        raise ImportError(f"Missing module '{module_name}'. Install it using: pip install {module_name}")
'''


    
class GoogleGeminiLLM(BaseLLM):
    """Wrapper for Google Gemini AI"""

    def __init__(self, model_name="gemini-pro", temperature=0.7):
        # ✅ Configure the API Key
        genai.configure(api_key=config.GOOGLE_GEMINI_API_KEY)
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """Generate text using Google Gemini API"""
        model = genai.GenerativeModel(self.model_name)
        
        response = model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )

        return response.text.strip() if response and hasattr(response, "text") else "No response from Gemini"


class OllamaLLM(BaseLLM):
    """Wrapper for Ollama LLM"""

    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content.strip()

class LLMSelector:
    """Factory to select the appropriate LLM provider"""
    
    @staticmethod
    def get_llm(provider: str, **kwargs) -> BaseLLM:
        providers = {
            "ollama": OllamaLLM,
            "google": GoogleGeminiLLM,
        }
        if provider.lower() not in providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return providers[provider.lower()](**kwargs)