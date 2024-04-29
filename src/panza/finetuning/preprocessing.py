import os
from typing import Dict

from panza.utils import prompting

SYSTEM_PREAMBLE_PATH = os.environ.get("PANZA_SYSTEM_PREAMBLE_PATH")
USER_PREAMBLE_PATH = os.environ.get("PANZA_USER_PREAMBLE_PATH")

SYSTEM_PREAMBLE = prompting.load_preamble(SYSTEM_PREAMBLE_PATH)
USER_PREAMBLE = prompting.load_user_preamble(USER_PREAMBLE_PATH)

PANZA_GENERATIVE_MODEL = os.environ.get("PANZA_GENERATIVE_MODEL")
PROMPT_START_WRAPPER, PROMPT_END_WRAPPER, RESPONSE_START_WRAPPER, RESPONSE_END_WRAPPER = (
    prompting.get_model_special_tokens(PANZA_GENERATIVE_MODEL)
)


r"""Example custom preprocessing function.

This is here to help illustrate the way to set up finetuning
on a local dataset. One step of that process is to create
a preprocessing function for your dataset, and that is what
is done below. Check out the LLM Finetuning section of
`../README.md` for more context.

For this example, we're going to pretend that our local dataset
is `./train.jsonl`.

Note: this dataset is actually a copy of one of our ARC-Easy
multiple-choice ICL eval datasets. And you would never actually
train on eval data! ... But this is just a demonstration.

Every example within the dataset has the format:
{
    'query': <query text>,
    'choices': [<choice 0 text>, <choice 1 text>, ...],
    'gold': <int> # index of correct choice
}

To enable finetuning, we want to turn this into a prompt/response
format. We'll structure prompts and responses like this:
{
    'prompt': <query text>\nOptions:\n - <choice 0 text>\n - <choice 1 text>\nAnswer: ,
    'response': <correct choice text>
}
"""


def panza_preprocessing_function(inp: Dict) -> Dict:
    try:
        prompt_raw = inp["summary"].split("\n\nInstruction: ")[-1]
        return {
            "prompt": PROMPT_START_WRAPPER + prompt_raw + PROMPT_END_WRAPPER,
            "response": RESPONSE_START_WRAPPER + inp["email"] + RESPONSE_END_WRAPPER,
        }
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e


def panza_preprocessing_function_train_with_preamble(inp: Dict) -> Dict:
    try:
        prompt_raw = inp["summary"].split("\n\nInstruction: ")[-1]
        prompt = prompting.create_prompt(prompt_raw, SYSTEM_PREAMBLE, USER_PREAMBLE)
        return {
            "prompt": PROMPT_START_WRAPPER + prompt + PROMPT_END_WRAPPER,
            "response": RESPONSE_START_WRAPPER + inp["email"] + RESPONSE_END_WRAPPER,
        }
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e
