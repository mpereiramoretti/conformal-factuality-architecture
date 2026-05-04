# =========================
# GPT-5 implementation
# =========================
import logging

from llms.llm_interface import LLMInterface

from openai import OpenAI


MODEL_NAME = "text-embedding-3-large"


class Text_Embedding_3_Large_Model(LLMInterface):

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
        max_tokens: int = 1000,
        n_samples: int = 1
    ) -> str:
        prompt = messages[0]["content"]
        try:
            response = self.client.embeddings.create(model=MODEL_NAME, input=prompt)
            return response.data[0].embedding
        except Exception as e:
            self.logger.warning(f"Model {MODEL_NAME} failed to generate: {e}")