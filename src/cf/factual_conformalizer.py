import logging
from math import ceil
from typing import List

import numpy as np
from tqdm import tqdm


class FactualConformalizer:
    def __init__(self, alpha, a, confidence_method):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.alpha = alpha
        self.a = a
        self.confidence_method = confidence_method
        self.threshold = None  # to be calibrated

    def get_threshold(self):
        if not self.threshold:
            raise ValueError("The threshold has not been calibrated yet. Do it by calling calibrate_threshold().")
        return self.threshold

    def calibrate_threshold(self, calibration_data: List[dict]):
        """
        Calibrates the quantile/threshold from conformal prediction (as in Hashimoto's paper).
        # self.alpha: float in (0, 1)
        # calibration_data: calibration data
        # self.a: required fraction correct
        # self.confidence_method: string
        """
        # Compute r score for each example.
        r_scores = [self.get_r_score(entry, self.confidence_method) for entry in calibration_data]

        # Compute threshold for conformal prection. The quantile is ceil((n+1)*(1-alpha))/n, and
        # We map this to the index by dropping the division by n and subtracting one (for zero-index).
        quantile_target_index = ceil((len(r_scores) + 1) * (1 - self.alpha))
        self.threshold = sorted(r_scores)[quantile_target_index - 1]
        self.logger.info(f"Calibrated threshold: {str(self.threshold)}")

    def get_r_score(self, entry: dict, confidence_method: str):
        """
        Compute the r_a score for entry when confidence_method is used as the sub-claim scoring function
        (as in Hashimoto's paper).
        """
        threshold_set = sorted(
            [
                subclaim[confidence_method] + subclaim["noise"]
                for subclaim in entry["claims"]
            ],
            reverse=True,
        )
        curr_threshold = threshold_set[0]
        for threshold in threshold_set:
            curr_threshold = threshold
            # Apply threshold.
            accepted_subclaims = [
                subclaim
                for subclaim in entry["claims"]
                if subclaim[confidence_method] + subclaim["noise"] >= threshold
            ]

            # Compute entailed/correct fraction.
            entailed_fraction = (
                np.mean(
                    [
                        subclaim["factual"] == 1
                        for subclaim in accepted_subclaims
                    ]
                )
                if accepted_subclaims
                else 1
            )

            if entailed_fraction < self.a:
                return curr_threshold
        return -100000  # -100000 is less than any score assigned by any of the implemented confidence methods

    def mark_as_accepted(self, entry: dict):
        for subclaim in entry["claims"]:
            subclaim["accepted"] = int(subclaim[self.confidence_method] >= self.threshold)
        self.logger.info(f"Done!")

    # for logging
    def log_data_stats(self, entry: dict):
        total_factual, total_subclaims, pct_factual = self.get_pct_factual(entry)
        total_accepted, total_subclaims, pct_accepted = self.get_pct_accepted(entry)
        msg = f"\nFactual subclaims: {total_factual} / {total_subclaims} ({pct_factual}%)"
        msg = msg + f"\nAccepted subclaims: {total_accepted} / {total_subclaims} ({pct_accepted}%)"
        msg = msg + "\n"
        self.logger.info(msg)

    def get_pct_accepted(self, entry: dict):
        return self.get_totals_and_pct(entry=entry, subset="accepted")

    def get_pct_factual(self, entry: dict):
        return self.get_totals_and_pct(entry=entry, subset="factual")

    def get_totals_and_pct(self, entry: dict, subset: str):
        total_subset = sum(subclaim[subset] for subclaim in entry["claims"])
        total_subclaims = len(entry["claims"])
        pct_subset = total_subset / total_subclaims * 100.0
        return total_subset, total_subclaims, pct_subset