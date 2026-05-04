import logging
from typing import List, Tuple

import numpy as np
from numpy import dot
from numpy.linalg import norm

from llms import llm_interface
from metrics.metric_interface import MetricInterface


class MergedCosineSimilarityMetric(MetricInterface):
    def __init__(self, embedding_model: llm_interface):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.embedding_model = embedding_model

    def get_name(self) -> str:
        return "merged-cosine-similarity-metric"

    def compute(self, entry: dict) -> float:
        if self.embedding_model:
            if entry["total_factual"] == 0:
                self.logger.warning("There are no factual subclaims in this entry... Returning 0.")
                return 0
            else:
                merged_factual = entry["merged_factual"]
                merged_controlled = entry["merged_accepted"]
                embedding_factual = self.get_embedding(merged_factual)
                embedding_controlled = self.get_embedding(merged_controlled)
                similarity = float(self.cosine_similarity(embedding_controlled, embedding_factual))
                return similarity
        else:
            raise ValueError("Not possible to compute cosine_similarity because no embedding model was passed.")

    def aggregate(self, data: dict) -> Tuple[float, float]:
        vals = []
        name = self.get_name()
        for entry in data:
            vals.append(entry[name])
        return (float(np.mean(vals)), float(np.std(vals)))

    def get_embedding(self, text: str) -> List[float]:
        messages = [{"role": "user", "content": text}]
        return self.embedding_model.generate(messages=messages)

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        return dot(a, b) / (norm(a) * norm(b))