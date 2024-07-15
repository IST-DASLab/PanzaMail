import argparse
import json
import time
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from panza.utils import rag


def load_inputs(path: str) -> List[Dict]:
    with open(path, "r") as f:
        lines = f.readlines()

    inputs = [json.loads(line) for line in lines]

    return inputs


def process_inputs(inputs: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Document]:
    # Convert inputs to langchain documents
    documents = [
        Document(page_content=text["text"])
        for text in inputs
    ]

    # Split long pieces of text into chuncks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(documents)

    return documents


def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Store the pieces of text in a embeddings vector DB.")
    parser.add_argument("--path-to-inputs", help="Path to the cleaned pieces of text as input")
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--chunk-overlap", type=int, default=3000)
    parser.add_argument("--db-path", type=str)
    parser.add_argument("--index-name", type=str)
    parser.add_argument(
        "--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2"
    )

    args = parser.parse_args()

    # Load the pieces of text
    texts = load_inputs(args.path_to_inputs)
    print(f"Loaded {len(texts)} pieces of text.")

    # Process the pieces of text
    documents = process_inputs(texts, args.chunk_size, args.chunk_overlap)
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
