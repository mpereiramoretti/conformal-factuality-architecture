import logging
from typing import Tuple

import numpy as np

from metrics.metric_interface import MetricInterface


class CorrelationCFMetric(MetricInterface):
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_name(self) -> str:
        return "correlation-cf-metric"

    def compute(self, entry: dict) -> float:
        pass

    def aggregate(self, data: dict) -> Tuple[float, float]:
        selected_entries = [entry for entry in data if entry["total_factual"] > 0]
        list_accepted_and_factual = [entry["total_accepted_and_factual"] for entry in selected_entries]
        list_accepted = [entry["total_accepted"] for entry in selected_entries]
        corr = float(np.abs(np.corrcoef(list_accepted_and_factual, list_accepted)[0, 1]))
        self.logger.info(list_accepted_and_factual)
        self.logger.info(list_accepted)
        self.logger.info(corr)
        return (corr, 0.0)