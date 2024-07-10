import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.evaluation import base_inference
from panza.utils import prompting, rag

sys.path.pop(0)


def main():
    parser = base_inference.get_base_inference_args_parser()
    args = parser.parse_args()

    print("Running inference with args:", args)

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

    while True:
        user_input = input("Enter another request  (or 'quit' to exit): ")

        if user_input.lower() == "quit":
            print("Exiting...")
            break

        prompts, outputs = base_inference.run_inference(
            instructions=[user_input],
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

        print("Processed input:", prompts[0])
        print("Generated text", outputs[0])


if __name__ == "__main__":
    main()
