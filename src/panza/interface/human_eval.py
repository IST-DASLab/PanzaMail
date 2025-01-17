from panza.writer import PanzaWriter
from panza.entities.instruction import EmailInstruction

from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

import csv, os, re


class PanzaHumanEval:
    def _response_generation_procedure(self, all_prompts, batch_size, responses_per_prompt):
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

    def generate_responses_to_prompts(self, all_prompts, batch_size, responses_per_prompt):
        return self._response_generation_procedure(all_prompts, batch_size, responses_per_prompt)

    def write_responses_to_csv(self, prompts, results):
        outputs = zip(prompts, results)
        if self.mode == "rating":
            data = [{"prompt": p, "email": r, "rating": None} for p, r in outputs]
            fieldnames = ["prompt", "email", "rating"]
        elif self.mode == "inference":
            data = [{"prompt": p, "email": r} for p, r in outputs]
            fieldnames = ["prompt", "email"]
        else:
            raise ValueError("This mode operation is not supported.")
        with open(
            self.output_file,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def generate_impersonation_study(self, batch_size, responses_per_prompt):
        if os.path.exists(self.output_file):
            # The output file exists, we simply need to fill in group 3 responses.
            prompts = []
            responses = []
            to_generate = []
            with open(self.output_file, "r") as f:
                lines = list(csv.reader(f))
            lines = lines[1:]  # Skip header
            for line in lines:
                prompt = line[0]
                prompts.append(prompt)
                response = line[1]
                if response != "TODO":
                    responses.append(line[1])
                else:
                    to_generate.append(prompt)
            responses.extend(
                self.generate_responses_to_prompts(to_generate, batch_size, responses_per_prompt)
            )
            return prompts, responses
        else:
            # The output file does not exist. Generate prompt groups and fill in groups 1 and 2.
            prompt_file = self.panza_workspace / "data" / "test.jsonl"
            with open(prompt_file, "r") as f:
                sampled_prompts = np.random.choice(
                    [json.loads(x) for x in f.readlines()], 30, replace=False
                )
            random_prompt_indices = np.arange(len(sampled_prompts))
            np.random.shuffle(random_prompt_indices)
            random_prompt_indices = list(random_prompt_indices)
            group_number = 3
            grouped_indices = [random_prompt_indices[x::group_number] for x in range(group_number)]
            prompts = []
            responses = []
            for idx, group_idxs in enumerate(grouped_indices):
                group_information = sampled_prompts[group_idxs]
                if idx == 0:
                    prompts.extend([(x["summary"]) for x in group_information])
                    responses.extend([(x["email"]) for x in group_information])
                elif idx == 1:
                    group_1_prompts = [x["summary"] for x in group_information]
                    prompts.extend(group_1_prompts)
                    responses.extend(
                        self.generate_responses_to_prompts(
                            group_1_prompts, batch_size, responses_per_prompt
                        )
                    )
                else:
                    prompts.extend([x["summary"] for x in group_information])
                    responses.extend(["TODO" for x in group_information])
            return prompts, responses

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
        path_to_fixed_prompt: str,
        mode: str,
        username: str,
    ):
        np.random.seed(seed)  # Fix seed to ensure that we always get the same prompts
        self.username = username
        self.panza_workspace = Path(panza_workspace)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.checkpoint = checkpoint
        self.eval_type = eval_type
        self.writer = writer
        self.mode = mode
        self.path_to_fixed_prompt = Path(path_to_fixed_prompt)

        if self.eval_type == "impersonation":
            self.output_file = self.output_folder / f"{self.username}_{self.eval_type}_eval.csv"
            prompts, results = self.generate_impersonation_study(batch_size, responses_per_prompt)
        else:
            self.output_file = (
                self.output_folder
                / f"{os.path.basename(self.checkpoint)}_{self.eval_type}_eval.csv"
            )
            # Used for User Studies 1 and 2.
            if self.eval_type == "fixed":
                prompt_file = self.panza_workspace / self.path_to_fixed_prompt
                with open(prompt_file, "r") as f:
                    prompts = [x.strip() for x in f.readlines()]
            # Used for User Study 2.
            elif self.eval_type == "own":
                prompt_file = self.panza_workspace / "data" / "test.jsonl"
                with open(prompt_file, "r") as f:
                    prompts = np.random.choice(
                        [json.loads(x)["summary"] for x in f.readlines()], 16, replace=False
                    )
            else:
                raise ValueError(
                    "This type of human evaluation is not supported. Please choose between fixed, own and impersonation."
                )
            results = self.generate_responses_to_prompts(prompts, batch_size, responses_per_prompt)
        self.write_responses_to_csv(prompts, results)
