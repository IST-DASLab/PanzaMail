import os
import sys

import ollama
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.evaluation import base_inference
from panza.utils import prompting, rag
from panza.utils.documents import Email

sys.path.pop(0)


def get_response_stream(prompt: str, model: str):
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    return stream


def print_response_stream(stream):
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)



def main():
    parser = base_inference.get_base_inference_args_parser()
    args = parser.parse_args()

    print("Running inference with args:", args)

    if args.nthreads is not None:
        torch.set_num_threads(args.nthreads)


    if args.use_rag:
        embeddings_model = rag.get_embeddings_model(args.embedding_model)
        db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

    system_preamble, user_preamble, rag_preamble, _ = prompting.load_all_preambles(
        args.system_preamble, args.user_preamble, args.rag_preamble, args.thread_preamble
    )

    while True:
        instruction = input("Enter another request  (or 'quit' to exit): ")

        if instruction.lower() == "quit":
            print("Exiting...")
            break

        relevant_emails = []
        if args.use_rag:
            assert db is not None, "RAG requires a database to be provided."
            re = db._similarity_search_with_relevance_scores(instruction, k=args.rag_num_emails)
            relevant_emails = [
                Email.deserialize(r[0].metadata["serialized_email"])
                for r in re
                if r[1] >= args.rag_relevance_threshold
            ]

        prompt = prompting.create_prompt(
            instruction,
            system_preamble,
            user_preamble,
            rag_preamble,
            relevant_emails,
        )

        print("Running with prompt:", prompt)

        args.model = "llama3.1"
        stream = get_response_stream(prompt, args.model)
        print_response_stream(stream)


if __name__ == "__main__":
    main()
