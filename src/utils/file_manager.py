import io
import json
import logging
from typing import List


class FileManager:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def load_input_lines(self, input_filename: str) -> List[str]:
        with io.open(input_filename, "r") as fopen:
            lines = fopen.readlines()
        return lines

    def save_output_to_file(self, data: dict, filename: str):
        with open(filename, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

    def load_output_from_file(self, filename: str) -> List[dict]:
        with open(filename, "r", encoding="utf-8") as fopen:
            return json.load(fopen)["data"]

    def head(self, lines: List[str], n: int = 5):
        if n > len(lines):
            n = len(lines)
            self.logger.info(f"Showing all the inputs:")
        else:
            self.logger.info(f"Showing the {n} leading inputs:")
        
        for i in range(n):
            self.logger.info(lines[i])