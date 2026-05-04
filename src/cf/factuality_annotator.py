import logging
import json
import re
from typing import List, Optional

from llms import llm_interface


class FactualityAnnotator:
    def __init__(self, model: llm_interface, factuality_prompt: str):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model = model
        self.factuality_prompt = factuality_prompt

    def get_annotations(self, entry: List[dict]) -> List[dict] | None:
        prompt = entry["prompt"]
        subclaims = [s["subclaim"] for s in entry["claims"]]
        string_subclaims = "\n".join(f"{i+1}. {subclaim}" for i, subclaim in enumerate(subclaims))
        prompt = self.factuality_prompt.\
            replace("<<PROMPT>>", prompt).\
            replace("<<SUBCLAIMS>>", string_subclaims)
        self.logger.info(prompt)
        messages = [{"role": "user", "content": prompt}]
        output = self.model.generate(messages=messages)
        self.logger.info(output)
        output = output.replace("```jsonl\n", "")
        output = output.replace("\\\"", "'")
        output = output.replace("\\", "\\\\")
        output = output.replace("```", "")
        return self.parse_llm_subclaims(output)

    def parse_llm_subclaims(self, output: str) -> Optional[List[str]]:
        # -------------------------
        # 1. Tentativa rápida (JSONL por linha)
        # -------------------------
        try:
            parsed = [
                json.loads(line)
                for line in output.splitlines()
                if line.strip().startswith("{")
            ]

            if parsed:
                return parsed

        except Exception as ex:
            self.logger.warning(f"Strict parse failed: {ex}")

        # -------------------------
        # 2. Fallback tolerante (regex)
        # -------------------------
        try:
            matches = re.findall(r'\{[^{}]*\}', output)

            parsed = []
            for m in matches:
                try:
                    obj = json.loads(m)
                    subclaim = obj.get("subclaim")

                    if isinstance(subclaim, str):
                        parsed.append(subclaim)

                except json.JSONDecodeError as ex:
                    self.logger.warning(f"Going to next line: {ex}")
                    continue

            if parsed:
                return parsed

        except Exception as ex:
            self.logger.warning(f"Loose parse failed: {ex}")

        # -------------------------
        # 3. Falha total
        # -------------------------
        self.logger.error("Final parsing failure")
        self.logger.error(output)

        return None