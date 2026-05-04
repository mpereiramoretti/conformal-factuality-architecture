# =========================
# Cocoruta 2.0 implementation
# =========================
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria

from llms.llm_interface import LLMInterface


MODEL_NAME = "felipeoes/cocoruta-2-8b"


class Cocoruta_2_0_Model(LLMInterface):

    def __init__(self, device_map="cuda:1"):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model = None
        self.tokenizer = None
        self.device_map = device_map
    
    def load_model(self):
        #bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.logger.info(f"Loading tokenizer for model {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map=self.device_map)
        self.logger.info(f"Loading model {MODEL_NAME}...")
        #self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=self.device_map, quantization_config=bnb_config)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=self.device_map, torch_dtype=torch.float16)
        self.logger.info("Finished loading the model!")

    def build_message(self, prompt):
        content = f"### Pergunta: {prompt}\n### Resposta:"
        return [{"role": "user", "content": content}]

    def generate(
        self,
        messages: list,  # always [{"role": "user", "content": prompt}] for Cocoruta 2.0
        temperature: float = 0,
        max_tokens: int = 1000,
        n_samples: int = 1
    ) -> str:

        # Define an early stopping criteria
        class StopOnString(StoppingCriteria):
            def __init__(self, target_sequence, prompt, tokenizer):
                self.target_sequence = target_sequence
                self.prompt=prompt
                self.tokenizer=tokenizer

            def __call__(self, input_ids, scores, **kwargs):
                # Get the generated text as a string
                generated_text = self.tokenizer.decode(input_ids[0])
                generated_text = generated_text.replace(self.prompt,'')
                # Check if the target sequence appears in the generated text
                if self.target_sequence in generated_text:
                    return True  # Stop generation

                return False  # Continue generation

        try:
            # INFERENCE
            stop_string = "###"
            input_text = messages[0]["content"]
            input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to(self.model.device)
            #if temperature == 0:
            #    temperature = 0.05  # zero instabilizes the model
            contents = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id, 
                max_new_tokens=max_tokens, 
                temperature=0.8,
                top_p=0.9,
                top_k=30,
                stopping_criteria=[StopOnString(stop_string, input_text, self.tokenizer)],
                do_sample=True,
                num_return_sequences=n_samples,
            )
        except Exception as e:
            self.logger.warning(f"Model {MODEL_NAME} failed to generate: {e}")

        if n_samples == 1:
            output = self.tokenizer.decode(contents[0], skip_special_tokens=True)
        else:
            output = [
                self.tokenizer.decode(content, skip_special_tokens=True)
                for content in contents
            ]

        if n_samples == 1:
            response = self.get_response(output)
        else:
            response = [self.get_response(x) for x in output]

        return response

    def get_response(self, output: str) -> str:
        try:
            response = output.split("### Resposta:")[1]
        except Exception as e:
            response = "Não foi possível responder..."
            self.logger.warning(f"Error while obtaining response for the prompt... Assigning default answer to question. Exception: {e}")
        return response
