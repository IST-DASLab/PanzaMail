import argparse
import os
import sys

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.utils import prompting

sys.path.pop(0)


def get_base_inference_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None)
    parser.add_argument("--system-preamble", type=str, default=None)
    parser.add_argument("--user-preamble", type=str, default=None)
    parser.add_argument("--rag-preamble", type=str, default=None)
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--nthreads", type=int, default=None)

    return parser


def load_model_and_tokenizer(model_path, device, dtype):
    assert dtype in [None, "fp32", "bf16"]
    if device == "cpu":
        assert dtype == "fp32", "CPU only supports fp32, please specify --dtype fp32"
    dtype = None if dtype is None else (torch.float32 if dtype == "fp32" else torch.bfloat16)

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("found an adapter.")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device
        )
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def run_inference(
    instruction,
    model,
    tokenizer,
    system_preamble,
    user_preamble,
    rag_preamble,
    rag_relevance_threshold,
    rag_num_emails,
    use_rag,
    db,
    max_new_tokens,
    best,
    temperature,
    top_k,
    top_p,
    device,
):
    relevant_emails = []
    if use_rag:
        relevant_emails = db._similarity_search_with_relevance_scores(instruction, k=rag_num_emails)
        relevant_emails = [r[0] for r in relevant_emails if r[1] >= rag_relevance_threshold]

    prompt = prompting.create_prompt(
        instruction, system_preamble, user_preamble, rag_preamble, relevant_emails
    )
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    if best:
        generated_ids = model.generate(
            model_inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
        )
    else:
        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    output = tokenizer.batch_decode(generated_ids)[0]
    return prompt, output
