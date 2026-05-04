from abc import ABC, abstractmethod


class LLMInterface(ABC):

    @abstractmethod
    def generate(
        self,
        messages: list,
        temperature: float = 0,
        max_tokens: int = 1000,
        n_samples: int = 1
    ) -> str:
        pass