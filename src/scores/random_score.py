import logging

import numpy as np

from scores.score_interface import ScoreInterface


class RandomScore(ScoreInterface):
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_name(self) -> str:
        return "random-score"

    def compute(self, entry: dict, subclaim: dict) -> float:
        return np.random.normal(0, 1)