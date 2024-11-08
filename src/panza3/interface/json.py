from panza3.entities.instruction import EmailInstruction
from panza3.writer import PanzaWriter

import json
import numpy as np
import os
import re
from tqdm import tqdm

from evaluate import load
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
import string

punc_table = str.maketrans({key: None for key in string.punctuation})
rouge = ROUGEScore()
bleu1 = BLEUScore(n_gram=1)
bleu2 = BLEUScore(n_gram=2)
bleu3 = BLEUScore(n_gram=3)
bleu4 = BLEUScore(n_gram=4)
mauve = load("mauve")


def compute_rouge_scores(predictions, goldens):
    goldens = [" ".join(x.translate(punc_table).lower().split()) for x in goldens]
    candidates = [
        " ".join(prediction.translate(punc_table).lower().split()) for prediction in predictions
    ]
    scores = [
        {k: v.item() for k, v in rouge(candidate, goldens).items()} for candidate in candidates
    ]
    return scores


def compute_bleu_scores(predictions, goldens):
    goldens = [" ".join(x.translate(punc_table).lower().split()) for x in goldens]
    candidates = [
        " ".join(prediction.translate(punc_table).lower().split()) for prediction in predictions
    ]
    bleu_scores = [
        np.mean([bleu([candidate], [goldens]) for bleu in [bleu1, bleu2, bleu3, bleu4]])
        for candidate in candidates
    ]
    return [s.item() for s in bleu_scores]


def compute_mauve_score(predictions, goldens):
    predictions = [
        prediction for nested_prediction in predictions for prediction in nested_prediction
    ]
    goldens = [golden for nested_golden in goldens for golden in nested_golden]
    mauve_score = mauve.compute(predictions=predictions, references=goldens)
    return mauve_score


class PanzaJSON:
    def compose_output_folder(self, json_path, checkpoint, panza_workspace, username):
        if os.path.isdir(checkpoint):
            # Presumably this is a Panza-trained model; go ahead
            # and put the json output into the same folder.
            output_dir = checkpoint
        else:
            # Assume that this is a huggingface model identified by its hf handle.
            # We don't want to populate the cached model folder, so instead
            # we create a folder in the Panza workspace to put the output.
            output_dir = os.path.join(
                panza_workspace, "checkpoints", "models", checkpoint, username
            )
            os.makedirs(output_dir, exist_ok=True)
        filename_no_ext = os.path.splitext(os.path.basename(json_path))[0]
        return os.path.join(output_dir, f"{filename_no_ext}_outputs.json")

    def assemble_responses(self, prompts_json, batch_size, use_thread, responses_per_prompt):

        with open(prompts_json, "r") as f:
            golden_lines = [json.loads(l) for l in f.readlines()]

        # Group json lines together by prompt to avoid weirdness in
        # eval metric computation. In case golden responses are provided,
        # all goldens are used as alternatives for BLEU and ROUGE scores;
        # the first one provided is used for MAUVE.
        grouped_golden = {}
        has_goldens = False
        for entry in golden_lines:
            # 'summary' is the name of the 'prompt' field, i.e., the one to group on.
            if entry["summary"] in grouped_golden:
                if "email" in entry:
                    has_goldens = True
                    grouped_golden[entry["summary"]]["goldens"].append(entry["email"])
            else:
                grouped_golden[entry["summary"]] = {}
                if "email" in entry:
                    has_goldens = True
                    grouped_golden[entry["summary"]]["goldens"] = [(entry["email"])]
            grouped_golden[entry["summary"]]["thread"] = entry["thread"]
        # Convert dict to list of (k, v) pairs to batch through it.
        grouped_golden = list(grouped_golden.items())

        all_responses = []
        for i in tqdm(range(0, len(grouped_golden), batch_size)):
            batch = grouped_golden[i : i + batch_size]
            prompts = [item[0] for item in batch]
            if use_thread:
                threads = [item[1]["thread"] for item in batch]
            golden_responses = [item[1]["goldens"] for item in batch]

            responses = [
                {
                    "prompt": p,
                    "full_prompt": None,
                    "thread": None if not use_thread else threads[i],
                    "golden_responses": golden_responses[i],
                    "panza_responses": [],
                }
                for i, p in enumerate(prompts)
            ]
            for _ in range(responses_per_prompt):
                if use_thread:
                    instructions = list(zip(prompts, threads))
                else:
                    instructions = list(zip(prompts, [None] * len(prompts)))

                outputs, full_prompts = self.writer.run_batch(
                    [EmailInstruction(user_input) for user_input in instructions],
                    return_prompt=True,
                )

                # Remove some boilerplate added by instruction-tuned models w/out finetuning.
                outputs = [o.replace("Here is the email:\n", "") for o in outputs]
                outputs = [re.sub(r"SUBJECT:.*\n", "", o) for o in outputs]
                outputs = [re.sub(r"Subject:.*\n", "", o) for o in outputs]
                outputs = [re.sub(r"E-MAIL CONTENT:.*\n", "", o) for o in outputs]

                for i, r in enumerate(responses):
                    r["full_prompt"] = full_prompts[i]
                    r["panza_responses"].append(outputs[i])
                all_responses += responses
        return all_responses, has_goldens

    def do_compute_metrics(self, all_responses):
        for response in all_responses:
            response["scores"] = {}
            response["scores"]["BLEU"] = compute_bleu_scores(
                response["panza_responses"], response["golden_responses"]
            )
            response["scores"]["ROUGE"] = compute_rouge_scores(
                response["panza_responses"], response["golden_responses"]
            )
        rouge_categories = all_responses[0]["scores"]["ROUGE"][0].keys()
        aggregate_metrics = {
            "BLEU": np.mean([s for r in all_responses for s in r["scores"]["BLEU"]]),
            "ROUGE": {
                cat: np.mean([s[cat] for r in all_responses for s in r["scores"]["ROUGE"]])
                for cat in rouge_categories
            },
            "MAUVE": compute_mauve_score(
                [r["panza_responses"] for r in all_responses],
                [r["golden_responses"] for r in all_responses],
            ).mauve,
        }
        print("########## Aggregated quality metrics ##########\n")
        print(json.dumps(aggregate_metrics, indent=2))
        return {"responses": all_responses, "aggregate_metrics": aggregate_metrics}

    def __init__(
        self,
        writer: PanzaWriter,
        checkpoint: str,
        panza_workspace: str,
        input_file: str,
        batch_size: int,
        use_thread: bool,
        responses_per_prompt: int,
        compute_metrics: bool,
        username: str,
    ):
        self.writer = writer
        responses, has_goldens = self.assemble_responses(
            input_file, batch_size, use_thread, responses_per_prompt
        )
        if compute_metrics:
            if has_goldens:
                responses = self.do_compute_metrics(responses)
            else:
                print(
                    "Warning: metrics requested but no golden labels given!",
                    "\nDumping responses without computing metrics.",
                )

        output_path = self.compose_output_folder(input_file, checkpoint, panza_workspace, username)
        with open(output_path, "w") as f:
            json.dump(responses, f, indent=4, sort_keys=True)
