import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Text

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from panza.utils import rag

sys.path.pop(0)


def retrieve_similar_texts(batch, db, num_texts):
    texts = []
    for text in batch:
        try:
            relevant_texts = db._similarity_search_with_relevance_scores(
                text["text"], k=num_texts
            )
        except Exception as e:
            print(f"Error in RAG search: {e}")
            relevant_texts = []
            return relevant_texts

        relevant_texts = [
            {"text": r[0].page_content, "score": r[1]}
            for r in relevant_texts
            if r[0].page_content not in text["text"]
        ]
        text["relevant_texts"] = relevant_texts
        texts.append(text)

    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Get similar pieces of text for Retrieval Augmented Fine Tuning (RAFT)"
    )
    parser.add_argument("--path-to-inputs", help="Path to the cleaned pieces of text as input")
    parser.add_argument(
        "--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rag-num-texts", type=int, default=7)
    args = parser.parse_args()

    assert args.path_to_inputs.endswith(
        ".jsonl"
    ), f"Expecting a .jsonl file, but given = {args.path_to_inputs}"

    print(f"--> Reading pieces of text from: {args.path_to_inputs}")

    # Read pieces of text
    with open(args.path_to_inputs, "r") as f:
        lines = f.readlines()
        json_lines = [json.loads(line.strip(",")) for line in lines]
        print(f"--> # pieces of text = {len(json_lines)}")

    embeddings_model = rag.get_embeddings_model(args.embedding_model)
    db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

    path_for_outputs = args.path_to_inputs.rsplit(".jsonl", 1)[0] + "_raft.jsonl"
    num_processed_inputs = 0
    start_time = time.time()
    with open(path_for_outputs, "w") as f:
        for i in tqdm(range(0, len(json_lines), args.batch_size)):
            # TODO(armand): Fix this print for batched inference
            print(f"--> Processing batch {i}/{len(json_lines)}")
            batch = json_lines[i : i + args.batch_size]
            texts = retrieve_similar_texts(batch, db, args.rag_num_texts)
            num_processed_inputs += len(texts)

            for item in texts:
                f.write(json.dumps(item))
                f.write("\n")

    elapsed_time = time.time() - start_time
    print(f"{elapsed_time:.2f} seconds to process {len(json_lines)} texts.")


if __name__ == "__main__":
    main()
