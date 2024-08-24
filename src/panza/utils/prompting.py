from typing import List, Optional, Text

from panza.utils.documents import Email

MISTRAL_PROMPT_START_WRAPPER = "[INST] "
MISTRAL_PROMPT_END_WRAPPER = " [/INST]"
MISTRAL_RESPONSE_START_WRAPPER = "<s>"
MISTRAL_RESPONSE_END_WRAPPER = "</s>"

LLAMA3_PROMPT_START_WRAPPER = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_PROMPT_END_WRAPPER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_RESPONSE_START_WRAPPER = ""
LLAMA3_RESPONSE_END_WRAPPER = "<|eot_id|>"

PHI3_PROMPT_START_WRAPPER = "<s><|user|> "
PHI3_PROMPT_END_WRAPPER = "<|end|><|assistant|> "
PHI3_RESPONSE_START_WRAPPER = ""
PHI3_RESPONSE_END_WRAPPER = "<|end|>"


def get_model_special_tokens(model_name):
    model_name = model_name.lower()
    if "llama" in model_name:
        prompt_start_wrapper = LLAMA3_PROMPT_START_WRAPPER
        prompt_end_wrapper = LLAMA3_PROMPT_END_WRAPPER
        response_start_wrapper = LLAMA3_RESPONSE_START_WRAPPER
        response_end_wrapper = LLAMA3_RESPONSE_END_WRAPPER
    elif "mistral" in model_name.lower():
        prompt_start_wrapper = MISTRAL_PROMPT_START_WRAPPER
        prompt_end_wrapper = MISTRAL_PROMPT_END_WRAPPER
        response_start_wrapper = MISTRAL_RESPONSE_START_WRAPPER
        response_end_wrapper = MISTRAL_RESPONSE_END_WRAPPER
    elif "phi" in model_name.lower():
        prompt_start_wrapper = PHI3_PROMPT_START_WRAPPER
        prompt_end_wrapper = PHI3_PROMPT_END_WRAPPER
        response_start_wrapper = PHI3_RESPONSE_START_WRAPPER
        response_end_wrapper = PHI3_RESPONSE_END_WRAPPER
    else:
        raise ValueError(f"Presets missing for prompting model {model_name}")

    return prompt_start_wrapper, prompt_end_wrapper, response_start_wrapper, response_end_wrapper
