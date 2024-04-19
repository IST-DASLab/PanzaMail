import argparse
import json
import os
import time
from typing import Dict, List, Text

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MDL = "mistralai/Mistral-7B-Instruct-v0.2"
TEMP = 0.7
TOP_P = 0.7
TOP_K = 50
MAX_TOKENS = 10000


class LLMSummarizer:
    def __init__(
        self, model, dtype, temperature, top_k, top_p, max_tokens, summarization_prompt
    ) -> None:
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype, device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.summarization_prompt = summarization_prompt

        # Save sampling parameters
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

    def prepare_batch_for_inference(self, emails: List[Dict]) -> List[Text]:
        batch_with_prompt = []
        for item in emails:
            prompt_with_email = self.summarization_prompt.format(email=item["email"])
            batch_with_prompt.append({"role": "user", "content": prompt_with_email})
        return batch_with_prompt

    def run_inference(self, emails: List[Dict]) -> List[Dict]:
        batch = self.prepare_batch_for_inference(emails)

        model_inputs = self.tokenizer.apply_chat_template(batch, return_tensors="pt")
        model_inputs = model_inputs.to(self.device)

        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        outputs = self.tokenizer.batch_decode(generated_ids)

        # Extract generated text
        summaries = []
        for output in outputs:
            output = output.split("[/INST]")[-1]
            output = output.split("</s>")[0]
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

        instruction = generated_text.split("Instruction: ", 1)[1]
        summarized_emails.append(
            {
                "email": emails[j]["email"],
                "subject": emails[j]["subject"],
                "summary": instruction,
            }
        )

    return summarized_emails


def main():
    parser = argparse.ArgumentParser(
        description="Transform emails into dataset for PANZA finetuning"
    )
    parser.add_argument("--path-to-emails", help="Path to the cleaned emails")
    parser.add_argument("--prompt-file", help="A path to file with prompt text")
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
    print(f"--> Processing with batch_size 1 and prompt = {summarization_prompt}")
    print(
        f"--> params for sampling:"
        f"\t model = {MDL}"
        f"\t temperature = {TEMP}"
        f"\t top_p = {TOP_P}"
        f"\t max_tokens = {MAX_TOKENS}"
    )

    # Read emails
    with open(args.path_to_emails, "r") as f:
        lines = f.readlines()
        json_lines = [json.loads(line) for line in lines]
        print(f"--> # emails = {len(json_lines)}")

    summarizer = LLMSummarizer(
        model=MDL,
        dtype=torch.bfloat16,
        temperature=TEMP,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        summarization_prompt=summarization_prompt,
    )

    # Generate synthetic instructions
    path_for_outputs = args.path_to_emails.rsplit(".jsonl", 1)[0] + "_summarized.jsonl"
    num_processed_emails = 0
    start_time = time.time()
    with open(path_for_outputs, "w") as f:
        for i in tqdm(range(0, len(json_lines), 1)):
            print(f"--> Processing batch {i}/{len(json_lines)}")
            batch = json_lines[i : i + 1]
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
