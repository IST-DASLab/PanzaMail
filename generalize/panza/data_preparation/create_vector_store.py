import argparse
import json
import time
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from panza.utils import rag


def load_emails(path: str) -> List[Dict]:
    with open(path, "r") as f:
        lines = f.readlines()

    emails = [json.loads(line) for line in lines]

    return emails


def process_emails(emails: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Document]:
    # Convert e-mails to langchain documents
    documents = [
        Document(page_content=email["email"], metadata={"subject": email["subject"]})
        for email in emails
    ]

    # Split long e-mails into text chuncks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(documents)

    return documents


def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Store emails in a embeddings vector DB.")
    parser.add_argument("--path-to-emails", help="Path to the cleaned emails")
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--chunk-overlap", type=int, default=3000)
    parser.add_argument("--db-path", type=str)
    parser.add_argument("--index-name", type=str)
    parser.add_argument(
        "--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2"
    )

    args = parser.parse_args()

    # Load emails
    emails = load_emails(args.path_to_emails)
    print(f"Loaded {len(emails)} emails.")

    # Process emails
    documents = process_emails(emails, args.chunk_size, args.chunk_overlap)
    print(f"Obtained {len(documents)} text chuncks.")

    # Initialize embeddings model
    embeddings_model = rag.get_embeddings_model(args.embedding_model)

    # Create vector DB
    print("Creating vector DB...")
    start = time.time()
    db = rag.create_vector_db(documents, embeddings_model)
    print(f"Vector DB created in {time.time() - start} seconds.")

    # Save vector DB to disk
    db.save_local(folder_path=args.db_path, index_name=args.index_name)
    print(f"Vector DB index {args.index_name} saved to {args.db_path}.")


if __name__ == "__main__":
    main()
