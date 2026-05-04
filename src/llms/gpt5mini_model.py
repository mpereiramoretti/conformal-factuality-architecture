# =========================
# GPT-5 implementation
# =========================
import logging
from typing import List

from llms.llm_interface import LLMInterface

from openai import OpenAI


MODEL_NAME = "gpt-5-mini"


class GPT5Mini_Model(LLMInterface):
    '''
    GPT 5 family has no 'temperature' nor 'n_samples' parameter. It is recomended to apply these treatments
    via prompt engineering: https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools
    '''

    def __init__(self, oai_key: str):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.oai_key = oai_key
        self.client = None
    
    def load_model(self):
        self.logger.info("Loading OpenAI client...")
        try:
            self.client = OpenAI(api_key=self.oai_key)
        except Exception as e:
            raise ValueError(f"Could not load the OpenAI client: {e}")
        self.logger.info("Successfully loaded the client!")

    def generate(
        self,
        messages: list,
        temperature: float = 0,
        max_tokens: int = 10000,
        n_samples: int = 1
    ) -> str | List[dict]:

        try:
            input_msg = messages[0]["content"]
            response = self.client.responses.create(
                model=MODEL_NAME,
                input=input_msg,
                max_output_tokens=max_tokens,
                text={"verbosity": "medium"},
                reasoning={"effort": "medium"},
                store=False,
            )
        except Exception as e:
            self.logger.warning(f"Model {MODEL_NAME} failed to generate: {e}")

        return response.output_text