import logging
from typing import List
from tqdm import tqdm

from cf.factual_conformalizer import FactualConformalizer
from llms import llm_interface


class SubclaimsMerger:
    def __init__(self, model: llm_interface, merge_prompt: str):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model = model
        self.merge_prompt = merge_prompt

    def generate_merged_outputs_loo(self, data, alpha, a, confidence_method):
        """
        Assigns new-output containing Leave-One-Out merged accepted subclaims
        (as in Hashimoto's paper).
        """
        self.logger.info(
            f"Merging accepted subclaims for a={a}, alpha={alpha} and confidence_method={confidence_method}"
        )
        cf = FactualConformalizer(alpha=alpha, a=a, confidence_method=confidence_method)
        for i in tqdm(range(len(data))):
            self.logger.info(f"CALIBRATION ENTRY: {i} - (ENTRY: {data[i]['index']})")
            calibration_data = data[:i] + data[i + 1 :]
            test_data = data[i]
            cf.calibrate_threshold(calibration_data)
            cf.mark_as_accepted(test_data)
            test_data["new-output"] = self.merge_accepted(test_data)
            self.logger.info(f"Output mergeado: {test_data['new-output']}")
            self.logger.info("")
            self.logger.info("-" * 80)
            self.logger.info("")

    def merge_accepted(self, entry: dict):
        prompt = entry["prompt"]
        subclaims = [subclaim for subclaim in entry["claims"] if subclaim["accepted"] == 1]
        return self.merge(prompt, subclaims)

    def merge_factual(self, entry: dict):
        prompt = entry["prompt"]
        subclaims = [subclaim for subclaim in entry["claims"] if subclaim["factual"] == 1]
        return self.merge(prompt, subclaims)

    def merge(self, prompt: str, subclaims: List[dict]) -> str:
        if subclaims:
            claim_string = "\n".join([str(i) + ": " + subclaim["subclaim"] for i, subclaim in enumerate(subclaims)])
            merge_prompt = self.merge_prompt.replace("{claim_string}", claim_string).replace("{prompt}", prompt)
            self.logger.info(merge_prompt)
            messages = [
                {
                    "role": "user",
                    "content": merge_prompt,
                },
            ]
            output = self.model.generate(messages=messages, max_tokens=1000)
        else:
            output = "Abstain."
        return output