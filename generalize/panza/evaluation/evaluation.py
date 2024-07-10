# We conduct evaluations with three scores.
# The BLEU score is frequently used to evaluate translations and compares n-grams in a 'golden'
#     translation to those in a candidate translation. Multiple golden translations are possible.
# The ROUGE score is frequently used for translation and summarization; it also looks at
#     n-gram similarity. It is actually several scores, since precision, recall, and F1 score are
#     reported separately.
# The MAUVE score measures distribution similarity (in the sense of KL-divergence) between the
#     targets and outputs, and is not computed on a per-example basis. The similarity is computed
#     in the latent space of an LLM, by default GPT-2.

import torch
import json
import os
import re
import string
import sys

from evaluate import load
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore

import numpy as np
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.evaluation import base_inference
from panza.utils import prompting, rag

sys.path.pop(0)


def main():
    parser = base_inference.get_base_inference_args_parser()
    parser.add_argument("--responses-per-prompt", type=int, default=1)
    parser.add_argument("--golden", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    args = parser.parse_args()
    
    rouge = ROUGEScore()
    # This library computes the BLEU score components separately. We do not use a length penalty.
    bleu1 = BLEUScore(n_gram=1)
    bleu2 = BLEUScore(n_gram=2)
    bleu3 = BLEUScore(n_gram=3)
    bleu4 = BLEUScore(n_gram=4)
    mauve = load('mauve')

    if args.nthreads is not None:
        torch.set_num_threads(args.nthreads)

    print("Loading model ", args.model)
    model, tokenizer = base_inference.load_model_and_tokenizer(args.model, args.device, args.dtype, load_in_4bit=args.load_in_4bit)

    if args.use_rag:
        embeddings_model = rag.get_embeddings_model(args.embedding_model)
        db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

    system_preamble, user_preamble, rag_preamble = prompting.load_all_preambles(
        args.system_preamble, args.user_preamble, args.rag_preamble
    )

    with open(args.golden, "r") as f:
        golden_lines = [json.loads(l) for l in f.readlines()]

    grouped_golden = {}
    for entry in golden_lines:
        if entry["summary"] in grouped_golden:
            grouped_golden[entry["summary"]]["templates"].append(entry["text"])
        else:
            grouped_golden[entry["summary"]] = {}
            grouped_golden[entry["summary"]]["templates"] = [(entry["text"])]

    print("Evaluating with batch size", args.batch_size)

    results = {}
    all_results = []
    prompt_scores = {}
    outputs_logs = {}
    grouped_golden = list(grouped_golden.items())
    for i in range(0, len(grouped_golden), args.batch_size):
        batch = grouped_golden[i:i + args.batch_size]
        prompts = [item[0] for item in batch]
        golden_responses = [item[1]["templates"] for item in batch]

        #prompt_scores = [[] for _ in range(len(prompts))]
        for _ in range(args.responses_per_prompt):
            full_prompts, outputs = base_inference.run_inference(
                instructions=prompts,
                model=model,
                tokenizer=tokenizer,
                system_preamble=system_preamble,
                user_preamble=user_preamble,
                rag_preamble=rag_preamble,
                rag_relevance_threshold=args.rag_relevance_threshold,
                rag_num_texts=args.rag_num_texts,
                use_rag=args.use_rag,
                db=db if args.use_rag else None,
                max_new_tokens=args.max_new_tokens,
                best=args.best,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
            )

            # Remove some boilerplate added by instruction-tuned models w/out finetuning.
            outputs = [o.replace("Here is the text:\n", "") for o in outputs]
            # TODO(sean): determine if these two lines for subject below are necessary
            outputs = [re.sub(r'SUBJECT:.*\n', "", o) for o in outputs]
            outputs = [re.sub(r'Subject:.*\n', "", o) for o in outputs]
            outputs = [re.sub(r'TEXT CONTENT:.*\n', "", o) for o in outputs]
            for j, prompt in enumerate(prompts):
                # We clean up the strings for the BLEU and ROUGE scores.
                punc_table = str.maketrans({key: None for key in string.punctuation})
                golden = [" ".join(x.translate(punc_table).lower().split()) for x in golden_responses[j]]
                candidate = " ".join(outputs[j].translate(punc_table).lower().split())
                
                rouge_score = rouge(outputs[j], golden_responses[j])
                bleu_score = np.mean([bleu([candidate], [golden]) for bleu in [bleu1, bleu2, bleu3, bleu4]])
                rouge_score = rouge(candidate, golden)
                if prompt not in prompt_scores.keys():
                    prompt_scores[prompt] = {"prompt": prompt, "full_prompt": full_prompts[j],
                                    "golden" : golden_responses[j],  "output": [outputs[j]],
                                    "BLEU": [bleu_score.item()]}
                    for score, value in rouge_score.items():
                        prompt_scores[prompt][score] = [value.item()]
                else:
                    prompt_scores[prompt]["output"].append(outputs[j])
                    prompt_scores[prompt]["BLEU"].append(bleu_score.item())
                    for score, value in rouge_score.items():
                        prompt_scores[prompt][score].append(value.item())

                print("\n-----------\n", "PROMPT:\n", prompt, "\n\nOUTPUT:\n", outputs[j], "\n\nBLEU SCORE:\n", bleu_score, "\n\nROUGE SCORE:\n", rouge_score)


    means = {}
    mins = {}
    score_names = [k for k in prompt_scores.values().__iter__().__next__().keys() if 'BLEU' in k or 'rouge' in k]

    for k in score_names:
        means[k] = np.mean([v for scores in prompt_scores.values() for v in scores[k] ])
        mins[k] = np.min([v for scores in prompt_scores.values() for v in scores[k] ])

    # To compute the MAUVE score, we need equal-length flat arrays of
    # outputs and goldens. If we have multiple outputs per prompt, we
    # output them all, with the same golden prompt.
    # TODO: not sure if it would be better to randomly sample from the
    # outputs in this case.
    # TODO: consider handling the case where there are also multiple golden
    # queries per output. (We don't use this for anything now).
    flattened_golden = []
    flattened_outputs = []
    for prompt_info in prompt_scores.values():
        flattened_golden += ([prompt_info["golden"][0]])*len(prompt_info['output'])
        flattened_outputs += prompt_info['output']
    mauve_score = mauve.compute(predictions=flattened_outputs, references=flattened_golden) 
    print("MAUVE score", mauve_score)
    means["MAUVE"] = mauve_score.mauve
    print("Mean scores across all prompts: ", {f"    {k}: {v}" for k, v in means.items()})


    # Optionally, update wandb run with eval scores
    rag_str = "RAG-" if args.use_rag else ""
    if args.wandb_run_id:
        with wandb.init(id=args.wandb_run_id, resume=True):
            wandb.log({f"EVAL/{k}-{rag_str}mean": v for k, v in means.items()})
            wandb.log({f"EVAL/{k}-{rag_str}min": v for k, v in mins.items()})
    else:
        print({f"EVAL/{k}-{rag_str}mean": v for k, v in means.items()})
        print({f"EVAL/{k}-{rag_str}min": v for k, v in mins.items()})

    with open(os.path.join(args.model, f"{rag_str}eval_responses.txt"), 'w') as f:
        json.dump(prompt_scores, f, ensure_ascii=False, indent=4)

    with open(os.path.join(args.model, f"{rag_str}eval_summary.txt"), 'w') as f:
        json.dump({"means": means, "mins": mins}, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
