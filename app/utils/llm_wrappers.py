from app.utils.llm import BaseLLM
from typing import Type
import sys

IMPORTS = {}
# Try importing each module individually and print specific errors if they fail
for module_name in ["openai", "anthropic", "cohere", "ollama"]:
    try:
        IMPORTS[module_name] = __import__(module_name)
    except ImportError:
        raise ImportError(f"Missing module '{module_name}'. Install it using: pip install {module_name}")



class OpenAILLM(BaseLLM):
    """Wrapper for OpenAI LLM"""
    
    def __init__(self, model_name="gpt-4", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response["choices"][0]["message"]["content"].strip()

class CohereLLM(BaseLLM):
    """Wrapper for Cohere AI"""
    
    def __init__(self, model_name="command-r", temperature=0.7):
        self.client = cohere.Client("YOUR_COHERE_API_KEY")
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature
        )
        return response.generations[0].text.strip()

class AnthropicLLM(BaseLLM):
    """Wrapper for Anthropic Claude"""

    def __init__(self, model_name="claude-2", temperature=0.7):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.completion.strip()

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
            "openai": OpenAILLM,
            "anthropic": AnthropicLLM,
            "cohere": CohereLLM,
            "ollama": OllamaLLM,    
        }
        if provider.lower() not in providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return providers[provider.lower()](**kwargs)