from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse, logging
import json, os
import numpy as np
np.random.seed(42)

import torch
import sys

sys.path.append(os.path.join(os.environ['PANZA_WORKSPACE'], 'src/panza/utils/'))

from preprocessing import panza_preprocessing_function_train_with_preamble
import prompting

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

parser = argparse.ArgumentParser()

parser.add_argument("--base-model", type=str, default=None)
parser.add_argument("--adapter", type=str, default=None)
parser.add_argument("--data-path", type=str, default="../train.jsonl")
parser.add_argument("--num-samples", type=int, default=8)
parser.add_argument("--system-preamble", type=str, default=None)
parser.add_argument("--user-preamble", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dtype", type=str, default="bf16")
parser.add_argument("--output-path", type=str, default=None)
parser.add_argument("--nbits", type=int, default=4)
args = parser.parse_args()

assert args.output_path is not None, 'specify --output-path'

pretrained_model_dir = args.base_model
rosa_path = args.adapter

system_preamble, user_preamble, rag_preamble = prompting.load_all_preambles(
    args.system_preamble, args.user_preamble, None
)
tokenizer = AutoTokenizer.from_pretrained(rosa_path, use_fast=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

with open(args.data_path, 'r') as f:
    lines = f.readlines()

selected_idx = np.random.choice(len(lines), size=args.num_samples)
calib_data = [json.loads(lines[i]) for i in selected_idx]
prompts = [panza_preprocessing_function_train_with_preamble(d) for d in calib_data]

examples = [tokenizer(prompt['prompt'] + prompt['response']) for prompt in prompts]

quantize_config = BaseQuantizeConfig(
    bits=args.nbits,
    group_size=128,
    desc_act=False,
)

tdtype = torch.bfloat16 if args.dtype == 'bf16' else (torch.float32 if args.dtype == 'fp32' else 'auto')
model = AutoGPTQForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantize_config,
    rosa_name_or_path=rosa_path,
    torch_dtype=tdtype,
    device_map=args.device,
    trust_remote_code=True
)

prefix = 'base_model.model.'
model.layers_block_name = prefix + model.layers_block_name
model.outside_layer_modules = [prefix + n for n in model.outside_layer_modules]

model.quantize(examples)

model.model = model.model.base_model.model
model.save_quantized(args.output_path)
tokenizer.save_pretrained(args.output_path)