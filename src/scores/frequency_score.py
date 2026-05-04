import json
import logging
from typing import List

from llms import llm_interface
from scores.score_interface import ScoreInterface


class FrequencyScore(ScoreInterface):
    def __init__(self,
                 specialist_model: llm_interface,
                 generalist_model: llm_interface,
                 counting_prompt: str,
                 n_samples: int):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.specialist_model = specialist_model
        self.generalist_model = generalist_model
        self.counting_prompt = counting_prompt
        self.n_samples = n_samples

    def get_name(self) -> str:
        return "frequency-score"

    def compute(self, prompt: str, subclaim: dict) -> float:
        # this score is different from the pattern
        pass

    def compute(self, entry: dict) -> List[int]:
        prompt = entry["prompt"]
        subclaims = [subclaim["subclaim"] for subclaim in entry["claims"]]
        messages = self.specialist_model.build_message(prompt=prompt)
        alternate_outputs = self.specialist_model.generate(messages=messages, n_samples=5)
        claim_string = "\n".join([str(i) + ": " + fact for i, fact in enumerate(subclaims)])
        final_scores = [0] * len(subclaims)
        for i, output in enumerate(alternate_outputs):
            self.logger.info(f"----- Generated Output {i + 1} -----")
            score_prompt = self.counting_prompt + claim_string + "\n\nO texto é:\n" + output
            self.logger.info(f"\nScore prompt:\n{score_prompt}")
            self.logger.info(repr(score_prompt))
            messages = [{"role": "user", "content": score_prompt}]
            content = self.generalist_model.generate(messages=messages)
            content = content.replace("```jsonl\n", "")
            content = content.replace("\\", "\\\\")
            content = content.replace("```", "")
            try:
                for line in content.splitlines():
                    scores = json.loads(line)
                    idx = int(scores["id"])
                    final_scores[idx] += float(scores["score"])
            except Exception as e:
                self.logger.error(f"Failed to parse as jsonl...\nContent: {content}\nException: {e}")
                final_scores = [0] * len(subclaims)
            self.logger.info(f"Final scores for this output: {str(final_scores)}")

        return final_scores