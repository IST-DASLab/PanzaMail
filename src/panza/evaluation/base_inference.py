import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.utils import prompting
from panza.utils.documents import Email
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

sys.path.pop(0)


def get_base_inference_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None)
    parser.add_argument("--system-preamble", type=str, default=None)
    parser.add_argument("--user-preamble", type=str, default=None)
    parser.add_argument("--rag-preamble", type=str, default=None)
    parser.add_argument("--thread-preamble", type=str, default=None)
    parser.add_argument("--best", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--use-rag", action="store_true", default=False)
    parser.add_argument("--rag-relevance-threshold", type=float, default=0.2)
    parser.add_argument(
        "--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--rag-num-emails", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--nthreads", type=int, default=None)
    parser.add_argument("--load-in-4bit", default=False, action="store_true")

    return parser


def load_model_and_tokenizer(model_path, device, dtype, load_in_4bit):

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("found an adapter.")
        if load_in_4bit:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path, device_map=device, quantization_config=quant_config, trust_remote_code=True
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, device_map=device, trust_remote_code=True
            )
        model = model.merge_and_unload()
    else:
        if load_in_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map=device, quantization_config=quant_config, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, device_map=device, trust_remote_code=True
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=model.config.max_position_embeddings
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_inference(
    instructions,
    model,
    tokenizer,
    system_preamble,
    user_preamble,
    rag_preamble,
    rag_relevance_threshold,
    rag_num_emails,
    thread_preamble,
    use_rag,
    db,
    max_new_tokens,
    best,
    temperature,
    top_k,
    top_p,
    device,
):
    batch = []
    prompts = []
    for instruction, thread in instructions:
        relevant_emails = []
        if use_rag:
            assert db is not None, "RAG requires a database to be provided."
            re = db._similarity_search_with_relevance_scores(
                instruction, k=rag_num_emails
            )
            relevant_emails = [
                Email.deserialize(r[0].metadata["serialized_email"])
                for r in re
                if r[1] >= rag_relevance_threshold
            ]

        prompt = prompting.create_prompt(
            instruction, system_preamble, user_preamble, rag_preamble, relevant_emails, thread_preamble, thread,
        )
        prompts.append(prompt)
        messages = [{"role": "user", "content": prompt}]
        batch.append(messages)
