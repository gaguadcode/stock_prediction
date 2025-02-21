from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract Base Class for LLM Wrappers"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM given a prompt"""
        pass