import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Text

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.utils import prompting

sys.path.pop(0)

MDL = os.environ.get("PANZA_GENERATIVE_MODEL")
TEMP = 0.7
TOP_P = 0.7
TOP_K = 50


class LLMSummarizer:
    def __init__(
        self, model, dtype, temperature, top_k, top_p, summarization_prompt, load_in_4bit
    ) -> None:
        self.device = "cuda"

        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quant_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map=self.device,
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            model_max_length=self.model.config.max_position_embeddings,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.summarization_prompt = summarization_prompt

        (
            _,
            self.prompt_end_wrapper,
            _,
            self.response_end_wrapper,
        ) = prompting.get_model_special_tokens(self.model.name_or_path)

        # Save sampling parameters
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def prepare_batch_for_inference(self, emails: List[Dict]) -> List[Text]:
        batch_with_prompt = []
        for item in emails:
            prompt_with_email = self.summarization_prompt.format(email=item["email"])
            batch_with_prompt.append([{"role": "user", "content": prompt_with_email}])
        return batch_with_prompt

    def run_inference(self, emails: List[Dict]) -> List[Dict]:
        gc.collect()
        torch.cuda.empty_cache()
        batch = self.prepare_batch_for_inference(emails)

        model_inputs = self.tokenizer.apply_chat_template(
            batch,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_dict=True,
        )
        model_inputs = model_inputs.to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        outputs = self.tokenizer.batch_decode(generated_ids)

        # Extract generated text
        summaries = []
        for output in outputs:
            output = output.split(self.prompt_end_wrapper)[-1]
            output = output.split(self.response_end_wrapper)[0]
            output = output.strip()
            summaries.append(output)

        return summaries


def generate_synthetic_instructions(emails: List[Dict], summarizer: LLMSummarizer):
    summarized_emails = []

    summaries = summarizer.run_inference(emails)

    for j, generated_text in enumerate(summaries):

        # Check if the outputs are valid
        keyword = "Instruction: "
        if generated_text.count(keyword) != 1:
            print(
                f"[WARNING] Skipping this sample:\n{generated_text}\n-----> "
                f"[REASON] it contains none or multiple instances of the keyword = {keyword}, "
                "but we expect exactly one"
            )
            continue

        instruction = generated_text.split(keyword, 1)[1]
        summarized_emails.append(
            {
                "email": emails[j]["email"],
                "subject": emails[j]["subject"],
                "summary": instruction,
                "thread": emails[j]["thread"],
                "date": emails[j]["date"],
            }
        )

    return summarized_emails


def main():
    parser = argparse.ArgumentParser(
        description="Transform emails into dataset for PANZA finetuning"
    )
    parser.add_argument("--path-to-emails", help="Path to the cleaned emails")
    parser.add_argument("--prompt-file", help="A path to file with prompt text")
    parser.add_argument("--batch-size", type=int, help="Inference batch size")
    parser.add_argument(
        "--load-in-4bit",
        default=False,
        action="store_true",
        help="Wheather to load the model in 4bit precision (BNB)",
    )
    parser.add_argument(
        "--fp32",
        default=False,
        action="store_true",
        help="Whether to use FP32 precision for computation",
    )
    args = parser.parse_args()

    assert args.path_to_emails.endswith(
        ".jsonl"
    ), f"Expecting a .jsonl file, but given = {args.path_to_emails}"

    assert os.path.exists(
        args.prompt_file
    ), f"Prompt file does not exist. Given path = {args.prompt_file}"
    with open(args.prompt_file, "r") as file:
        summarization_prompt = file.read()

    print(f"--> Reading emails from: {args.path_to_emails}")
    print(f"--> Processing with batch_size {args.batch_size} and prompt = {summarization_prompt}")
    print(
        f"--> params for sampling:"
        f"\t model = {MDL}"
        f"\t temperature = {TEMP}"
        f"\t top_p = {TOP_P}"
    )

    # Read emails
    with open(args.path_to_emails, "r") as f:
        lines = f.readlines()
        json_lines = [json.loads(line.strip(",")) for line in lines]
        print(f"--> # emails = {len(json_lines)}")

    summarizer = LLMSummarizer(
        model=MDL,
        dtype=torch.float32 if args.fp32 else torch.bfloat16,
        temperature=TEMP,
        top_p=TOP_P,
        top_k=TOP_K,
        summarization_prompt=summarization_prompt,
        load_in_4bit=args.load_in_4bit,
    )

    # Generate synthetic instructions
    path_for_outputs = args.path_to_emails.rsplit(".jsonl", 1)[0] + "_summarized.jsonl"
    num_processed_emails = 0
    start_time = time.time()
    with open(path_for_outputs, "w") as f:
        for i in tqdm(range(0, len(json_lines), args.batch_size)):
            # TODO(armand): Fix this print for batched inference
            print(f"--> Processing batch {i}/{len(json_lines)}")
            batch = json_lines[i : i + args.batch_size]
            summarized_emails = generate_synthetic_instructions(batch, summarizer)
            num_processed_emails += len(summarized_emails)

            # Write the summarized emails to a file
            for item in summarized_emails:
                f.write(json.dumps(item))
                f.write("\n")

    elapsed_time = time.time() - start_time
    print(f"{elapsed_time:.2f} seconds to process {len(json_lines)} emails.")


if __name__ == "__main__":
    main()
