# =========================
# Model Factory
# =========================
import logging

from llms import llm_interface
from llms.gpt5mini_model import GPT5Mini_Model
from llms.cocoruta_2_0_model import Cocoruta_2_0_Model
from llms.text_embedding_3_large_model import Text_Embedding_3_Large_Model


class ModelFactory:
    def __init__(self, oai_key: str, device_map: str = "cuda:1"):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.oai_key = oai_key
        self.device_map = device_map

    def get_model(self, model_name: str) -> llm_interface:
        if model_name == "gpt-5-mini":
            return GPT5Mini_Model(oai_key=self.oai_key)
        elif model_name == "felipeoes/cocoruta-2-8b":
            return Cocoruta_2_0_Model(device_map=self.device_map)
        elif model_name == "text-embedding-3-large":
            return Text_Embedding_3_Large_Model(oai_key=self.oai_key)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")