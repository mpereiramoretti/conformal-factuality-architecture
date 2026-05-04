import logging
import re

import numpy as np

from llms import llm_interface
from scores.score_interface import ScoreInterface


class AverageModelScore(ScoreInterface):
    def __init__(self, model: llm_interface, scoring_prompt: str, iterations: int = 5):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model = model
        self.scoring_prompt = scoring_prompt
        self.iterations = iterations

    def get_name(self) -> str:
        return "average-model-score"

    def compute(self, entry: dict, subclaim: dict) -> float:
        scores = [0.0] * self.iterations
        for i in range(self.iterations):
            try:
                replaced_prompt = self.scoring_prompt.replace("{claim}", subclaim["subclaim"])
                messages = self.model.build_message(replaced_prompt)
                output = self.model.generate(messages=messages)
                score = self.extract_first_number(output)
            except Exception as e:
                self.logger.warning(f"Error scoring prompt. Returning 0.0. Exception: {e}")
                score = 0.0
            self.logger.info(f"Iteration {i + 1}: {str(score)}")
            scores[i] = score
        self.logger.info(f"Model Scores: {str(scores)}")
        return np.mean(scores)

    def extract_first_number(self, text: str) -> float:
        match = re.search(r"\b\d+[.,]\d+\b", text)
        if not match:
            raise ValueError("There wasn't a number in the format x,xx or x.xx in the output string.")
        first_match = float(match.group().replace(",", "."))
        if (first_match > 1.0) or (first_match < 0.0):
            raise ValueError(f"Could not cast a number in the given format (got {str(first_match)} wrongly from the output string).")
        return first_match