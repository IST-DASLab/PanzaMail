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
from panza.utils.documents import Email

sys.path.pop(0)


def retrieve_similar_emails(batch, db, num_emails):
    emails = []
    for email in batch:
        try:
            relevant_emails = db._similarity_search_with_relevance_scores(
                email["email"], k=num_emails
            )
        except Exception as e:
            print(f"Error in RAG search: {e}")
            relevant_emails = []
            return relevant_emails

        relevant_emails = [
            {"serialized_email": r[0].metadata["serialized_email"], "score": r[1]}
            for r in relevant_emails
            if r[0].page_content not in email["email"]
        ]
        email["relevant_emails"] = relevant_emails
        emails.append(email)

    return emails


def main():
    parser = argparse.ArgumentParser(
        description="Get similar emails for Retrieval Augmented Fine Tuning (RAFT)"
    )
    parser.add_argument("--path-to-emails", help="Path to the cleaned emails")
    parser.add_argument(
        "--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rag-num-emails", type=int, default=7)
    args = parser.parse_args()

    assert args.path_to_emails.endswith(
        ".jsonl"
    ), f"Expecting a .jsonl file, but given = {args.path_to_emails}"

    print(f"--> Reading emails from: {args.path_to_emails}")

    # Read emails
    with open(args.path_to_emails, "r") as f:
        lines = f.readlines()
        json_lines = [json.loads(line.strip(",")) for line in lines]
        print(f"--> # emails = {len(json_lines)}")

    embeddings_model = rag.get_embeddings_model(args.embedding_model)
    db = rag.load_vector_db_from_disk(args.db_path, args.index_name, embeddings_model)

    path_for_outputs = args.path_to_emails.rsplit(".jsonl", 1)[0] + "_raft.jsonl"
    num_processed_emails = 0
    start_time = time.time()
    with open(path_for_outputs, "w") as f:
        for i in tqdm(range(0, len(json_lines), args.batch_size)):
            # TODO(armand): Fix this print for batched inference
            print(f"--> Processing batch {i}/{len(json_lines)}")
            batch = json_lines[i : i + args.batch_size]
            emails = retrieve_similar_emails(batch, db, args.rag_num_emails)
            num_processed_emails += len(emails)

            for item in emails:
                f.write(json.dumps(item))
                f.write("\n")

    elapsed_time = time.time() - start_time
    print(f"{elapsed_time:.2f} seconds to process {len(json_lines)} emails.")


if __name__ == "__main__":
    main()
