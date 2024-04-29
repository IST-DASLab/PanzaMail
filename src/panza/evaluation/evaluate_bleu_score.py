import json
import os
import sys

import nltk
import numpy as np
import torch
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
            grouped_golden[entry["summary"]]["templates"].append(entry["email"])
        else:
            grouped_golden[entry["summary"]] = {}
            if "prompt_type" not in entry:
                grouped_golden[entry["summary"]]["prompt_type"] = "natural"
            else:
                grouped_golden[entry["summary"]]["prompt_type"] = entry["prompt_type"]
            grouped_golden[entry["summary"]]["templates"] = [(entry["email"])]

    print("Evaluating with batch size", args.batch_size)

    results = {}
    outputs_logs = {}
    grouped_golden = list(grouped_golden.items())
    for i in range(0, len(grouped_golden), args.batch_size):
        batch = grouped_golden[i:i + args.batch_size]
        prompts = [item[0] for item in batch]
        prompt_types = [item[1]["prompt_type"] for item in batch]
        golden_responses = [item[1]["templates"] for item in batch]

        prompt_scores = [[] for _ in range(len(prompts))]
        for _ in range(args.responses_per_prompt):
            full_prompts, outputs = base_inference.run_inference(
                instructions=prompts,
                model=model,
                tokenizer=tokenizer,
                system_preamble=system_preamble,
                user_preamble=user_preamble,
                rag_preamble=rag_preamble,
                rag_relevance_threshold=args.rag_relevance_threshold,
                rag_num_emails=args.rag_num_emails,
                use_rag=args.use_rag,
                db=db if args.use_rag else None,
                max_new_tokens=args.max_new_tokens,
                best=args.best,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
            )

            for i, prompt in enumerate(prompts):
                bleu_score = nltk.translate.bleu_score.sentence_bleu(golden_responses[i], outputs[i])
                result = {"prompt": prompt, "full_prompt": full_prompts[i],
                                    "golden" : golden_responses[i][0],  "output": outputs[i],
                                    "bleu_score": bleu_score}
                prompt_scores[i].append(result)
                print("\n-----------\n", "PROMPT:\n", result["prompt"], "\n\nOUTPUT:\n", result["output"], "\n\nBLEU SCORE:\n", result["bleu_score"])

        for i, prompt_type in enumerate(prompt_types):
            if prompt_type in results.keys():
                results[prompt_type].append(np.mean([x["bleu_score"] for x in prompt_scores[i]]))
                for score in prompt_scores[i]:
                    outputs_logs[prompt_type].append(score)
            else:
                results[prompt_type] = [np.mean([x["bleu_score"] for x in prompt_scores[i]])]
                outputs_logs[prompt_type] = []
                for score in prompt_scores[i]:
                    outputs_logs[prompt_type].append(score)

    final_results = []
    for prompt_type, bleu_scores in results.items():
        final_results.append([prompt_type, np.mean(bleu_scores), np.min(bleu_scores)])

    print(final_results)

    # Optionally, update wandb run with BLEU scores
    if args.wandb_run_id:
        rag_str = "RAG" if args.use_rag else ""
        with wandb.init(id=args.wandb_run_id, resume=True):
            for prompt_type, bleu_scores_mean, bleu_scores_min in final_results:
                wandb.log({f"BLEU/{prompt_type}/BLEU-{rag_str}-mean": bleu_scores_mean})
                wandb.log({f"BLEU/{prompt_type}/BLEU-{rag_str}-min": bleu_scores_min})

    with open(os.path.join(args.model, f"{'rag_' if args.use_rag else ''}eval_responses.txt"), 'w') as f:
        #f.writelines(["\t".join([str(x) for x in score]) for result in outputs.values() for score in result])
        json.dump(outputs_logs, f, ensure_ascii=False, indent=4)

    with open(os.path.join(args.model, f"{'rag_' if args.use_rag else ''}eval_summary.txt"), 'w') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
