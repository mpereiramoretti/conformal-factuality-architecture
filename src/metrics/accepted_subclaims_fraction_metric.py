import logging
from typing import Tuple

import numpy as np

from metrics.metric_interface import MetricInterface


class AcceptedSubclaimsFractionMetric(MetricInterface):
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_name(self) -> str:
        return "accepted-subclaims-fraction-metric"

    def compute(self, entry: dict) -> float:
        if entry["total_factual"] == 0:
            self.logger.warning("There are no factual subclaims in this entry... Returning 0.")
            return 0
        else:
            total_accepted_and_factual = entry['total_accepted_and_factual']
            total_factual = entry['total_factual']
            self.logger.info(f"total_accepted_and_factual = {total_accepted_and_factual} ; total_factual = {total_factual}")
            return float(np.round(entry["total_accepted_and_factual"] / entry["total_factual"], 6))

    def aggregate(self, data: dict) -> Tuple[float, float]:
        vals = []
        name = self.get_name()
        for entry in data:
            vals.append(entry[name])
        return (float(np.mean(vals)), float(np.std(vals)))