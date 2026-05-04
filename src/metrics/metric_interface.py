from abc import ABC, abstractmethod
from typing import Tuple


class MetricInterface(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def compute(self, entry: dict) -> float:
        pass

    @abstractmethod
    def aggregate(self, data: dict) -> Tuple[float, float]:  # (value, std)
        pass