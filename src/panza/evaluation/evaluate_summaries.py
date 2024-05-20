# We conduct evaluations with three scores.
# The BLEU score is frequently used to evaluate translations and compares n-grams in a 'golden'
#     translation to those in a candidate translation. Multiple golden translations are possible.
# The ROUGE score is frequently used for translation and summarization; it also looks at
#     n-gram similarity. It is actually several scores, since precision, recall, and F1 score are
#     reported separately.
# The MAUVE score measures distribution similarity (in the sense of KL-divergence) between the
#     targets and outputs, and is not computed on a per-example basis. The similarity is computed
#     in the latent space of an LLM, by default GPT-2.


import argparse
import copy
import json
import os
import re
import string
import sys
import time
from tqdm import tqdm
from typing import Dict, List

from evaluate import load
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)

from panza.data_preparation.summarize_emails import LLMSummarizer

TEMP = 0.7
TOP_P = 0.7
TOP_K = 50



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
                "golden_summary": emails[j]["golden_summary"],
                # "subject": emails[j]["subject"],  # TODO(armand): Handle subject metadata
                "summary": instruction,
            }
        )

    return summarized_emails


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--summarized-emails-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--golden-loc", default=None,
                         help="If not given, emails in summarized-emails-file are used as golden labels.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch-size", type=int, help="Inference batch size")
    parser.add_argument("--load-in-4bit", default=False, action="store_true")
    parser.add_argument("--prompt-file", help="A path to file with prompt text")
    parser.add_argument("--use-email-as-golden", help="use email as the golden summary", action="store_true")
    args = parser.parse_args()

    assert args.golden_loc is None or args.golden_loc.endswith(
        ".jsonl"
    ), f"Expecting a .jsonl file, but given = {args.path_to_emails}"

    assert (args.model is not None) or (args.golden_loc is not None and args.summarized_emails_file is not None
                                        ), "Either args.model should be given, or both of golden_loc and summarized_emails_file should be given."


    golden_loc = args.golden_loc or args.summarized_emails_file
    print("golden_loc is", golden_loc)
    with open(golden_loc, "r") as f:
        lines = f.readlines()
        golden_summaries = [json.loads(line.strip(',')) for line in lines]
        print(f"--> # emails = {len(golden_summaries)}")

    if args.use_email_as_golden or not args.golden_loc:
        def add_email_as_golden(x):
            y = copy.deepcopy(x)
            y["golden_summary"] = x["email"]
            return(y)
        golden_summaries = [add_email_as_golden(x) for x in golden_summaries]
    else:
        # Rename the summary field to "golden_summary" if necessary
        def maybe_rename_summary(k):
            if k == "summary":
                return "golden_summary"
            return k
        golden_summaries = [{maybe_rename_summary(k):v for k, v in golden.items()} for golden in golden_summaries]
    

    if args.model is not None:
        llm_summaries = golden_summaries
        summarized_emails = []

        assert os.path.exists(
            args.prompt_file
        ), f"Prompt file does not exist. Given path = {args.prompt_file}"
        with open(args.prompt_file, "r") as file:
            summarization_prompt = file.read()

        summarizer = LLMSummarizer(
            model=args.model,
            dtype=torch.float32 if args.dtype=="float32" else torch.bfloat16,
            temperature=TEMP,
            top_p=TOP_P,
            top_k=TOP_K,
            summarization_prompt=summarization_prompt,
            load_in_4bit=args.load_in_4bit
        )

        print(f"Running analysis on {len(llm_summaries)} emails.")
        for i in tqdm(range(0, len(llm_summaries), args.batch_size)):
            batch = llm_summaries[i : i + args.batch_size]
            summarized_emails +=generate_synthetic_instructions(batch, summarizer)
  
    else:
        assert args.golden_loc is not None and args.summarized_emails_file is not None
        with open(args.summarized_emails_file, "r") as f:
            lines = f.readlines()
            llm_summaries = [json.loads(line.strip(',')) for line in lines]
        def find_golden(email):
            for golden_email in golden_summaries:
                if email["email"] ==  golden_email["email"]:
                    new_email = copy.deepcopy(email)
                    new_email["golden_summary"] = golden_email["golden_summary"]
                    return new_email
            return None
        summarized_emails = [find_golden(email) for email in llm_summaries if find_golden(email) is not None]
        print(f"matched {len(summarized_emails)} emails")

    # Now we measure the quality!
    rouge = ROUGEScore()
    # This library computes the BLEU score components separately. We do not use a length penalty.
    bleu1 = BLEUScore(n_gram=1)
    bleu2 = BLEUScore(n_gram=2)
    bleu3 = BLEUScore(n_gram=3)
    bleu4 = BLEUScore(n_gram=4)

    punc_table = str.maketrans({key: None for key in string.punctuation})
    results = []
    for email in summarized_emails:
        print(email)
        golden = [" ".join(email["golden_summary"].translate(punc_table).lower().split())]
        candidate = " ".join(email["summary"].translate(punc_table).lower().split())

        bleu_score = np.mean([bleu([candidate], [golden]) for bleu in [bleu1, bleu2, bleu3, bleu4]])
        rouge_score = rouge(candidate, golden)
        new_email = copy.deepcopy(email)
        new_email["bleu_score"] = bleu_score.item()
        for score, value in rouge_score.items():
            new_email[score] = value.item()
        results.append(new_email)


    metric_means = {}
    for score in results.__iter__().__next__().keys():
        if 'bleu' in score or 'rouge' in score:
            metric_means[score] = np.mean([r[score] for r in results])
    print(metric_means)

    if args.model is not None:
        model_str = 'llama' if 'Llama' in args.model else 'mistral'
        outfile = f"{os.environ.get('PANZA_WORKSPACE')}/data/summarization_results/{golden_loc}-{model_str}/"
    else:
        outfile = args.summarized_emails_file[:-len(".jsonl")] + "/"
    print("See results at", outfile)
    os.makedirs(outfile, exist_ok=True)
    with open(outfile + "summarization_results.txt", 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    with open(outfile + "summarization_summary.txt", 'w') as f:
        json.dump(metric_means, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
