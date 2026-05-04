import logging
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from cf.factual_conformalizer import FactualConformalizer
from metrics.metric_interface import MetricInterface


class FactualConformalizerPlotter:
    def __init__(self, dataset_prefix):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.dataset_prefix = dataset_prefix

    def get_title_name(self):
        if self.dataset_prefix == "cocoruta":
            return "Cocoruta 2.0"
        return self.dataset_prefix

    def get_legend_name(self, confidence_method):
        if confidence_method == "baseline-score":
            return "Ordinal"
        elif confidence_method == "random-score":
            return "Random"
        elif confidence_method == "model-score":
            return "Cocoruta 2.0 confidence"
        elif confidence_method == "average-model-score":
            return "Cocoruta 2.0 average confidence"
        elif confidence_method == "frequency-score":
            return "Frequency"
        elif confidence_method == "factual":
            return "Optimal"

    def plot_hist(
        self,
        threshold,
        a,
        alpha,
        confidence_method,
        data,
        hist_filename
    ):
        """
        Creates histogram showing the fraction of subclaims removed across all test outputs.
        """
        self.logger.info(f"Creating histogram for threshold={threshold} (a={a}, alpha={alpha}, confidence_method={confidence_method})")
        plt.figure(dpi=800)
        fraction_removed_list = []
        for entry in tqdm(data):
            fraction_removed = 1 - (entry["total_accepted"] / entry["total_subclaims"])
            fraction_removed_list.append(fraction_removed)

        fontsize = 15
        plt.subplots(figsize=(6, 3.5))
        plt.xlabel("Percent removed", fontsize=fontsize)
        plt.ylabel("Fraction of outputs", fontsize=fontsize)
        plt.title(
            f"{self.get_title_name()}, threshold={threshold:.6f} ({chr(945)}={alpha})", fontsize=fontsize
        )
        weights = np.ones_like(fraction_removed_list) / float(len(fraction_removed_list))
        plt.hist(fraction_removed_list, weights=weights)
        plt.savefig(hist_filename, bbox_inches="tight", dpi=800)

    def plot_hist_loo(
        self,
        a,
        alpha,
        confidence_method,
        data,
        hist_filename
    ):
        """
        Creates histogram showing the fraction of subclaims removed across all calibration outputs (Leave-One-Out)
        (as in Hashimoto's paper).
        """
        self.logger.info(
            f"Creating histogram for a={a}, alpha={alpha} and confidence_method={confidence_method}"
        )
        plt.figure(dpi=800)
        fraction_removed_list = []
        cf = FactualConformalizer(alpha=alpha, a=a, confidence_method=confidence_method)
        for i in tqdm(range(len(data))):
            calibration_data = data[:i] + data[i + 1 :]
            test_data = data[i]
            cf.calibrate_threshold(calibration_data)
            threshold = cf.get_threshold()

            accepted_subclaims = [
                subclaim
                for subclaim in test_data["claims"]
                if subclaim[confidence_method] + subclaim["noise"] >= threshold
            ]
            fraction_removed = 1 - len(accepted_subclaims) / len(test_data["claims"])
            fraction_removed_list.append(fraction_removed)

        fontsize = 15
        plt.subplots(figsize=(6, 3.5))
        plt.xlabel("Percent removed", fontsize=fontsize)
        plt.ylabel("Fraction of outputs", fontsize=fontsize)
        plt.title(
            f"{self.get_title_name()}, {chr(945)}={alpha}", fontsize=fontsize
        )
        weights = np.ones_like(fraction_removed_list) / float(len(fraction_removed_list))
        plt.hist(fraction_removed_list, weights=weights)
        plt.savefig(hist_filename, bbox_inches="tight", dpi=800)

    def create_correctness_vs_removed_plot_loo(
        self,
        data,
        alphas,
        a,
        confidence_methods,
        fig_filename
    ):
        """
        Creates leave-one-out conformal plots (as in Hashimoto's paper).
        """
        self.logger.info(f"Producing conformal plot: {fig_filename}")
        plt.figure(dpi=800)

        for confidence_method in tqdm(confidence_methods):
            results = []  # first indexes into alpha, then list of (correct, frac_removed)_i

            for alpha in alphas:
                results_for_alpha = [[], []]
                
                cf = FactualConformalizer(
                    alpha=alpha,
                    a=a,
                    confidence_method=confidence_method
                )
                for i in range(len(data)):
                    calibration_data = data[:i] + data[i + 1 :]
                    test_data = data[i]

                    cf.calibrate_threshold(calibration_data)
                    threshold = cf.get_threshold()
                    accepted_subclaims = [
                        subclaim
                        for subclaim in test_data["claims"]
                        if subclaim[confidence_method] + subclaim["noise"]
                        >= threshold
                    ]
                    fraction_removed = 1 - len(accepted_subclaims) / len(
                        test_data["claims"]
                    )
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
                    correctness = entailed_fraction >= a
                    results_for_alpha[0].append(correctness)
                    results_for_alpha[1].append(fraction_removed)

                results.append(results_for_alpha)

            x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
            y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]

            # Add standard error.
            yerr = [
                np.std(results_for_alpha[1]) * 1.96 / np.sqrt(len(results_for_alpha[1]))
                for results_for_alpha in results
            ]
            label = self.get_legend_name(confidence_method)

            plt.errorbar(x, y, yerr=yerr, label=label, linewidth=2)

        # Plot base factuality point.
        x_point = x[-1]
        y_point = y[-1]
        point_size = 235
        plt.scatter(
            x_point,
            y_point,
            color="black",
            marker="*",
            s=point_size,
            label="Base factuality",
            zorder=1000,
        )

        font_size = 16
        legend_font_size = 13
        dataset_title_name = self.get_title_name()
        if a == 1:
            plt.title(dataset_title_name, fontsize=font_size + 4)
            plt.xlabel("Fraction of factual outputs", fontsize=font_size)
        else:
            plt.title(f"{dataset_title_name}, a={a}", fontsize=font_size + 4)
            plt.xlabel(f"Fraction achieving avg factuality >= {a}")
        plt.ylabel("Average percent removed", fontsize=font_size)

        legend = plt.legend(
            loc="upper left", bbox_to_anchor=(1, 1), fontsize=legend_font_size
        )
        legend.get_title().set_fontsize(legend_font_size)
        plt.savefig(fig_filename, bbox_inches="tight")


    def create_calibration_plot(
        self,
        data,
        alphas,
        a,
        confidence_method,
        fig_filename
    ):
        """
        Creates calibration plot (as in Hashimoto's paper).
        """
        self.logger.info(f"Producing calibration plot: {fig_filename}")
        plt.subplots(figsize=(6, 4))

        results = []  # first indexes into alpha. then list of (correct, frac_removed)_i

        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            cf = FactualConformalizer(
                alpha=alpha,
                a=a,
                confidence_method=confidence_method
            )
            for i in range(1000):
                # Randomly shuffle the data
                random.shuffle(data)

                # Split the data into two equal parts
                split_index = len(data) // 2
                calibration_data = data[:split_index]
                test_data = data[split_index:]

                cf.calibrate_threshold(calibration_data)
                threshold = cf.get_threshold()

                accepted_subclaim_list = [
                    [
                        subclaim
                        for subclaim in test_data_point["claims"]
                        if subclaim[confidence_method] + subclaim["noise"]
                        >= threshold
                    ]
                    for test_data_point in test_data
                ]
                entailed_fraction_list = [
                    (
                        np.mean(
                            [
                                subclaim["factual"] == 1
                                for subclaim in accepted_subclaims
                            ]
                        )
                        if accepted_subclaims
                        else 1
                    )
                    for accepted_subclaims in accepted_subclaim_list
                ]
                correctness_list = [
                    entailed_fraction >= a for entailed_fraction in entailed_fraction_list
                ]
                fraction_correct = sum(correctness_list) / len(correctness_list)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)

            results.append(results_for_alpha)

        x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
        y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]
        # yerr = [np.std(results_for_alpha[1]) for results_for_alpha in results]

        self.logger.info(x)
        self.logger.info(y)
        # plt.fill_between(np.array(x), np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), color="#ADD8E6")

        x_values = np.linspace(0.3, 0.98, 100)

        # Plot lower bound.
        y_values = x_values
        plt.plot(
            x_values, y_values, "--", color="gray", linewidth=2, label="Thrm 3.1 bounds"
        )

        # Plot upper bound
        y_values = x_values + 1 / (split_index + 1)
        plt.plot(x_values, y_values, "--", color="gray", linewidth=2)
        plt.plot(x, y, label=self.get_title_name(), linewidth=2)

        plt.xlabel(f"Target factuality (1 - {chr(945)})", fontsize=16)
        plt.legend()
        plt.ylabel("Empirical factuality", fontsize=16)
        plt.savefig(fig_filename, bbox_inches="tight", dpi=800)

    def plot_metrics_distribution(self, results, k, fig_filename):
        """
        Plot boxplots of "_mean" metrics across folds.

        Parameters
        ----------
        results : dict
            Dictionary where each key is a fold (e.g., "Fold 1") and each value is
            another dict containing metrics, e.g.:
            {
                "Fold 1": {
                    "a-metric_mean": float,
                    "a-metric_std": float,
                    ...
                },
                ...
            }
            Only keys ending with "_mean" are used.

        k : int
            Number of folds (used in the plot title).

        Returns
        -------
        None
        """
        metrics = {}

        for fold_data in results.values():
            for key, value in fold_data.items():
                if key.endswith("_mean"):
                    metrics.setdefault(key, []).append(value)

        labels = list(metrics.keys())
        data = [metrics[label] for label in labels]
        labels = [label.replace("-metric_mean", "") for label in metrics.keys()]

        plt.figure()
        plt.boxplot(data, labels=labels)

        plt.xticks(rotation=45, ha="right")
        plt.title(f"Metrics distribution by fold (K = {k})")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(fig_filename, bbox_inches="tight", dpi=800)