import logging
import json
import re
from typing import List, Optional

from llms import llm_interface


class SubclaimsDecomposer:
    def __init__(self, model: llm_interface, breakdown_prompt: str):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model = model
        self.breakdown_prompt = breakdown_prompt

    def decompose_into_subclaims(self, entry: dict) -> List[str]:
        return self.get_subclaims(output=entry["output"])

    def get_subclaims(self, output: str) -> List[str] | None:
        """
        Takes in an output text and breaks it down into a list of sub-claims.
        """
        # Break into sub-claims.
        messages = [
            {
                "role": "user",
                "content": self.breakdown_prompt + output,
            },
        ]
        self.logger.info(self.breakdown_prompt + output)
        output = self.model.generate(messages=messages)
        self.logger.info(output)
        output = output.replace("```jsonl\n", "")
        output = output.replace("\\", "\\\\")
        output = output.replace("```", "")
        subclaims_list = self.parse_llm_subclaims(output)
        return subclaims_list

    def parse_llm_subclaims(self, output: str) -> Optional[List[str]]:
        """
        Faz parse de uma saída de LLM esperando JSONL com {"subclaim": "..."}.
        Tolera texto extra antes/depois e linhas inválidas.

        Params:
            output: string retornada pelo LLM
            logger: opcional (ex: self.logger)

        Returns:
            Lista de subclaims ou None se falhar completamente
        """

        # -------------------------
        # 1. Tentativa rápida (JSONL por linha)
        # -------------------------
        try:
            parsed = [
                json.loads(line)["subclaim"]
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