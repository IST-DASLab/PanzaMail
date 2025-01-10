from panza.writer import PanzaWriter
from panza.entities.instruction import EmailInstruction

from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

import csv, os, re


class PanzaHumanEval:
    def generate_responses_to_prompts(self, all_prompts, batch_size, responses_per_prompt):
        all_responses = []
        for i in tqdm(range(0, len(all_prompts), batch_size)):
            prompts = all_prompts[i : i + batch_size]
            for _ in range(responses_per_prompt):
                instructions = list(zip(prompts, [[]] * len(prompts)))

                responses = self.writer.run_batch(
                    [
                        EmailInstruction(user_input[0], thread=user_input[1])
                        for user_input in instructions
                    ],
                    return_prompt=False,
                )
                # Remove some boilerplate added by instruction-tuned models w/out finetuning.
                responses = [o.replace("Here is the email:\n", "") for o in responses]
                responses = [re.sub(r"SUBJECT:.*\n", "", o) for o in responses]
                responses = [re.sub(r"Subject:.*\n", "", o) for o in responses]
                responses = [re.sub(r"E-MAIL CONTENT:.*\n", "", o) for o in responses]
                all_responses += responses
        return all_responses

    def write_responses_to_csv(self, prompts, results):
        data = [{"prompt": p, "email": r, "rating": None} for p, r in zip(prompts, results)]
        with open(
            self.output_folder / f"{os.path.basename(self.checkpoint)}_{self.eval_type}_eval.csv",
            "w",
            newline="",
        ) as f:
            fieldnames = ["prompt", "email", "rating"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def __init__(
        self,
        writer: PanzaWriter,
        panza_workspace: str,
        checkpoint: str,
        responses_per_prompt: int,
        output_folder: str,
        seed: int,
        eval_type: str,
        batch_size: int,
    ):
        np.random.seed(seed)  # Fix seed to ensure that we always get the same prompts
        panza_workspace = Path(panza_workspace)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.checkpoint = checkpoint
        self.eval_type = eval_type

        if self.eval_type == "fixed":
            prompt_file = panza_workspace / "src" / "panza" / "evaluation" / "fixed_prompts.txt"
            with open(prompt_file, "r") as f:
                prompts = [x.strip() for x in f.readlines()]
        elif self.eval_type == "own":
            prompt_file = panza_workspace / "data" / "test.jsonl"
            with open(prompt_file, "r") as f:
                prompts = np.random.choice(
                    [json.loads(x)["summary"] for x in f.readlines()], 16, replace=False
                )
        else:
            raise ValueError(
                "This type of human evaluation is not supported. Please choose between fixed and own."
            )
        results = self.generate_responses_to_prompts(prompts, batch_size, responses_per_prompt)
        self.write_responses_to_csv(prompts, results)
