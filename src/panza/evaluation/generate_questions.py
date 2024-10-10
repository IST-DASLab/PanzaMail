# We conduct evaluations with three scores.
# The BLEU score is frequently used to evaluate translations and compares n-grams in a 'golden'
#     translation to those in a candidate translation. Multiple golden translations are possible.
# The ROUGE score is frequently used for translation and summarization; it also looks at
#     n-gram similarity. It is actually several scores, since precision, recall, and F1 score are
#     reported separately.
# The MAUVE score measures distribution similarity (in the sense of KL-divergence) between the
#     targets and outputs, and is not computed on a per-example basis. The similarity is computed
#     in the latent space of an LLM, by default GPT-2.

import csv
import os
import re
import string
import sys
import json

import numpy as np
np.random.seed(0) # Fix the seed so we always get the same prompts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.evaluation import base_inference
from panza.utils import prompting, rag

sys.path.pop(0)


def run_batch_inference(prompts, model, tokenizer, batch_size, system_preamble, user_preamble, rag_preamble,
                        rag_relevance_threshold, rag_num_emails, use_rag, db, max_new_tokens, best, temperature, top_k, top_p, device):
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        print(i)
        batch = prompts[i:i+batch_size]
        instructions = list(zip(batch, [None]*len(batch)))
        _, outputs = base_inference.run_inference(
            instructions=instructions,
            model=model,
            tokenizer=tokenizer,
            system_preamble=system_preamble,
            user_preamble=user_preamble,
            rag_preamble=rag_preamble,
            rag_relevance_threshold=rag_relevance_threshold,
            rag_num_emails=rag_num_emails,
            thread_preamble="",
            use_rag=use_rag,
            db=db if use_rag else None,
            max_new_tokens=max_new_tokens,
            best=best,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        outputs = [o.replace("Here is the email:\n", "") for o in outputs]
        outputs = [re.sub(r'SUBJECT:.*\n', "", o) for o in outputs]
        outputs = [re.sub(r'Subject:.*\n', "", o) for o in outputs]
        outputs = [re.sub(r'E-MAIL CONTENT:.*\n', "", o) for o in outputs]
        all_outputs += outputs
    return all_outputs






def run_eval(prompts, model, tokenizer, system_preamble,
             user_preamble, rag_preamble, thread_preamble, db, args, repeat_first=False):
    prompt_scores = []
    response_scores = []
    if args.use_thread:
        threads = [item[1]["thread"] for item in prompts]

    for _ in range(args.responses_per_prompt):
        if args.use_thread:
            instructions = list(zip(prompts, threads))
        else:
            instructions = list(zip(prompts, [None]*len(prompts)))

        _, outputs = base_inference.run_inference(
            instructions=instructions,
            model=model,
            tokenizer=tokenizer,
            system_preamble=system_preamble,
            user_preamble=user_preamble,
            rag_preamble=rag_preamble,
            rag_relevance_threshold=args.rag_relevance_threshold,
            rag_num_emails=args.rag_num_emails,
            thread_preamble=thread_preamble,
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
        outputs = [o.replace("Here is the email:\n", "") for o in outputs]
        outputs = [re.sub(r'SUBJECT:.*\n', "", o) for o in outputs]
        outputs = [re.sub(r'Subject:.*\n', "", o) for o in outputs]
        outputs = [re.sub(r'E-MAIL CONTENT:.*\n', "", o) for o in outputs]
        
        # Compute prompt and response BLEU scores
        bleu_scores_prompts = []
        bleu_scores_responses = []
        for j, promptj in enumerate(prompts):
            for i, prompti in enumerate(prompts):
                if i == j:
                    continue
                punc_table = str.maketrans({key: None for key in string.punctuation})
                golden = [" ".join(prompti.translate(punc_table).lower().split())]
                candidate = " ".join(promptj.translate(punc_table).lower().split())
                bleu_score = np.mean([bleu([candidate], [golden]) for bleu in [bleu1, bleu2, bleu3, bleu4]])
                bleu_scores_prompts.append(bleu_score)
                golden = [" ".join(outputs[i].translate(punc_table).lower().split())]
                candidate = " ".join(outputs[j].translate(punc_table).lower().split())
                bleu_score = np.mean([bleu([candidate], [golden]) for bleu in [bleu1, bleu2, bleu3, bleu4]])
                bleu_scores_responses.append(bleu_score)
        prompt_scores.append(np.mean(bleu_scores_prompts))
        response_scores.append(np.mean(bleu_scores_responses))

    return  [prompt_scores, response_scores]

def main():
    parser = base_inference.get_base_inference_args_parser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out-path")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--test-data-file", default=None)
    args = parser.parse_args()

    if args.prompts_file is not None:
        with open(args.prompts_file, 'r') as f:
            prompts = [x.strip() for x in f.readlines()]
            eval_style = "fixed_prompt"
    elif args.test_data_file is not None:
        with open(args.test_data_file, 'r') as f:
            prompts = np.random.choice([json.loads(x)["summary"] for x in f.readlines()], 16)
            eval_style = "own_prompt"
    else:
        raise ValueError("no prompts file given!")

    print("Loading model ", args.model)
    model, tokenizer = base_inference.load_model_and_tokenizer(args.model, args.device, args.dtype, load_in_4bit=args.load_in_4bit)

    if args.use_rag:
        embeddings_model = rag.get_embeddings_model(args.embedding_model)
        db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

    db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

    system_preamble, user_preamble, rag_preamble, _ = prompting.load_all_preambles(
        args.system_preamble, args.user_preamble, args.rag_preamble, args.thread_preamble
    )

    results = run_batch_inference(prompts, model, tokenizer, args.batch_size, system_preamble, user_preamble, rag_preamble,
                        args.rag_relevance_threshold, args.rag_num_emails, args.use_rag, db, args.max_new_tokens,
                        args.best, args.temperature, args.top_k, args.top_p, args.device)
    
    data = [{"prompt": p, "email": r, "rating": None} for p, r in zip(prompts, results)]
    print(len(data))
    with open(os.path.join(args.out_path, f"{os.path.basename(args.model)}_{eval_style}_eval.csv"), 'w', newline="") as f:
        fieldnames = ['prompt', 'email', 'rating']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    main()
