import os
import random
from typing import Dict

from langchain_core.documents import Document

from panza.utils import prompting, rag

SYSTEM_PREAMBLE_PATH = os.environ.get("PANZA_SYSTEM_PREAMBLE_PATH")
USER_PREAMBLE_PATH = os.environ.get("PANZA_USER_PREAMBLE_PATH")

SYSTEM_PREAMBLE = prompting.load_preamble(SYSTEM_PREAMBLE_PATH)
USER_PREAMBLE = prompting.load_user_preamble(USER_PREAMBLE_PATH)

PANZA_GENERATIVE_MODEL = os.environ.get("PANZA_GENERATIVE_MODEL")
PROMPT_START_WRAPPER, PROMPT_END_WRAPPER, RESPONSE_START_WRAPPER, RESPONSE_END_WRAPPER = (
    prompting.get_model_special_tokens(PANZA_GENERATIVE_MODEL)
)

PANZA_FINETUNE_WITH_RAG = int(os.environ.get("PANZA_FINETUNE_WITH_RAG")) == 1
if PANZA_FINETUNE_WITH_RAG:
    EMBEDDINGS_MODEL = os.environ.get("PANZA_EMBEDDING_MODEL")
    DB_PATH = os.environ.get("PANZA_DATA_DIR")
    INDEX_NAME = os.environ.get("PANZA_USERNAME")
    EMBEDDINGS_MODEL = rag.get_embeddings_model(EMBEDDINGS_MODEL)
    DB = rag.load_vector_db_from_disk(DB_PATH, INDEX_NAME, EMBEDDINGS_MODEL)
    RAG_PREAMBLE_PATH = os.environ.get("PANZA_RAG_PREAMBLE_PATH")
    RAG_PREAMBLE = prompting.load_preamble(RAG_PREAMBLE_PATH)
    RAG_NUM_TEXTS = int(os.environ.get("PANZA_FINETUNE_RAG_NUM_TEXTS"))
    RAG_PROB = float(os.environ.get("PANZA_FINETUNE_RAG_PROB"))
    RAG_RELEVANCE_THRESHOLD = float(os.environ.get("PANZA_FINETUNE_RAG_RELEVANCE_THRESHOLD"))
    PANZA_SEED = int(os.environ.get("PANZA_SEED"))
    random.seed(PANZA_SEED)

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


def filter_relevant_texts(relevant_texts):
    # Random chance to not include any relevant texts
    p = random.random()
    if p > RAG_PROB:
        relevant_texts = []
        print("Skip RAG")
        return relevant_texts

    if not relevant_texts:
        print("Relevant texts not found.")
        return []

    print("Don't skip")
    relevant_texts = [r["text"] for r in relevant_texts if r["score"] >= RAG_RELEVANCE_THRESHOLD]
    relevant_texts = [Document(page_content=text, metadata={}) for text in relevant_texts]
    relevant_texts = relevant_texts[:RAG_NUM_TEXTS]
    print(f"Found {len(relevant_texts)} relevant pieces of texts.")
    return relevant_texts


def panza_preprocessing_function(inp: Dict) -> Dict:
    try:
        prompt_raw = inp["summary"].split("\n\nInstruction: ")[-1]
        return {
            "prompt": PROMPT_START_WRAPPER + prompt_raw + PROMPT_END_WRAPPER,
            "response": RESPONSE_START_WRAPPER + inp["text"] + RESPONSE_END_WRAPPER,
        }
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e


def panza_preprocessing_function_train_with_preamble(inp: Dict) -> Dict:
    try:
        prompt_raw = inp["summary"].split("\n\nInstruction: ")[-1]
        if PANZA_FINETUNE_WITH_RAG:
            relevant_texts = inp.get("relevant_texts", [])
            relevant_texts = filter_relevant_texts(relevant_texts)
            prompt = prompting.create_prompt(prompt_raw, SYSTEM_PREAMBLE, USER_PREAMBLE, RAG_PREAMBLE, relevant_texts)
            print(prompt)
        else:
            prompt = prompting.create_prompt(prompt_raw, SYSTEM_PREAMBLE, USER_PREAMBLE)
        return {
            "prompt": PROMPT_START_WRAPPER + prompt + PROMPT_END_WRAPPER,
            "response": RESPONSE_START_WRAPPER + inp["text"] + RESPONSE_END_WRAPPER,
        }
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e
