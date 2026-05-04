import logging

from scores.score_interface import ScoreInterface


class BaselineScore(ScoreInterface):
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_name(self) -> str:
        return "baseline-score"

    def compute(self, entry: dict, subclaim: dict) -> float:
        idx = -1
        for i, item in enumerate(entry["claims"]):
            if item["subclaim"] == subclaim["subclaim"]:
                idx = i
                break
        return len(entry["claims"]) - (idx + 1)