import argparse
import os
import sys

import torch
from peft import AutoPeftModelForCausalLM
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

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
    parser.add_argument("--rag-num-texts", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--nthreads", type=int, default=None)
    parser.add_argument("--load-in-4bit", default=False, action="store_true")

    return parser


def load_model_and_tokenizer(model_path, device, dtype, load_in_4bit):
    assert dtype in [None, "fp32", "bf16"]
    if device == "cpu":
        assert dtype == "fp32", "CPU only supports fp32, please specify --dtype fp32"
    dtype = None if dtype is None else (torch.float32 if dtype == "fp32" else torch.bfloat16)

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if load_in_4bit
        else None
    )

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
    rag_num_texts,
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
    for instruction in instructions:
        relevant_texts = []
        if use_rag:
            relevant_texts = db._similarity_search_with_relevance_scores(
                instruction, k=rag_num_texts
            )
            relevant_texts = [r[0] for r in relevant_texts if r[1] >= rag_relevance_threshold]

        prompt = prompting.create_prompt(
            instruction, system_preamble, user_preamble, rag_preamble, relevant_texts
        )
        prompts.append(prompt)
        messages = [{"role": "user", "content": prompt}]
        batch.append(messages)

    encodeds = tokenizer.apply_chat_template(
        batch,
        return_tensors="pt",
        add_generation_prompt=True,
        padding=True,
        truncation=True,
        return_dict=True,
    )
    model_inputs = encodeds.to(device)

    if best:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    outputs = tokenizer.batch_decode(generated_ids)

    # Clean outputs
    _, prompt_end_wrapper, _, response_end_wrapper = prompting.get_model_special_tokens(
        model.name_or_path
    )
    outputs = [
        output.split(prompt_end_wrapper)[-1].split(response_end_wrapper)[0] for output in outputs
    ]

    return prompts, outputs
