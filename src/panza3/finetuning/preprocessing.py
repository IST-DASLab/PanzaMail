import os
from typing import Dict

import hydra
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from panza3.entities import EmailInstruction

PREPROCESSING_CONFIG_FILE = os.environ.get("PANZA_PREPROCESSING_CONFIG")
if PREPROCESSING_CONFIG_FILE:
    print("Hello from preprocessing.py")

    preprocessing_config = OmegaConf.load(PREPROCESSING_CONFIG_FILE)
    prompt_builder = hydra.utils.instantiate(preprocessing_config.prompting)

    # Load tokenizer
    config = AutoConfig.from_pretrained(preprocessing_config.model)
    tokenizer = AutoTokenizer.from_pretrained(
        preprocessing_config.model, model_max_length=config.max_position_embeddings
    )


def panza_preprocessing_function(inputs: Dict) -> Dict:
    try:
        prompt_raw = inputs["summary"].split("\n\nInstruction: ")[-1]
        instruction = EmailInstruction(instruction=prompt_raw, thread=inputs.get("thread", []))
        prompt = prompt_builder.build_prompt(instruction)

        print(f"Prompt: {prompt}")

        # Generate the full conversation
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": inputs["email"]}
        ]
        chat_prompt = tokenizer.apply_chat_template(conversation, tokenize=False)

        # Identify the index where the response begins
        response_begin_index = chat_prompt.index(inputs["email"])

        # Split the full prompt into prompt and response
        prompt = chat_prompt[:response_begin_index]
        response = chat_prompt[response_begin_index:]

        return {
            "prompt": prompt,
            "response": response,
        }
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inputs}") from e
