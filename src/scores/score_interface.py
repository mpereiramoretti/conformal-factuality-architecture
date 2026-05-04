from abc import ABC, abstractmethod

from scipy.stats import rankdata


class ScoreInterface(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def compute(self, entry: dict, subclaim: dict) -> float:
        pass

    #def compute_ranking_scores(self, subclaims_list: List[dict], subclaim: dict) -> int:
    #    confidence_method = self.get_name()
    #    score_list = [
    #        -(subclaim[confidence_method] + subclaim["noise"])
    #        for subclaim in entry["claims"]
    #    ]
    #    rankings = len(entry["claims"]) + 1 - rankdata(score_list, method="ordinal")
    #    rankings = rankings / len(entry["claims"])
    #    return rankings