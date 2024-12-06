import datetime
import json
import logging
import os
import random
import shutil
import time
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from panza import PanzaWriter  # The import also loads custom Hydra resolvers
from panza.entities import Document, Email, SummarizationInstruction
from panza.retriever import DocumentRetriever
from panza.data_preparation.extract_emails import extract_emails
from panza.data_preparation.rag import create_vector_store

LOGGER = logging.getLogger(__name__)


def rename_config_keys(cfg: DictConfig) -> None:
    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    cfg.writer.llm.sampling_parameters = cfg.writer.llm.sampling
    del cfg.writer.llm.sampling

    cfg.writer.prompt_builder = cfg.writer.prompting
    del cfg.writer.prompting

    # Re-enable struct mode to lock down the configuration
    OmegaConf.set_struct(cfg, True)


def load_documents(data_path: str) -> None:
    assert data_path.endswith(".jsonl"), f"Expecting a .jsonl file, but given = {data_path}"

    LOGGER.info(f"--> Reading emails from: {data_path}")

    with open(data_path, "r") as f:
        lines = f.readlines()
    documents = [Email.deserialize(line.strip(",")) for line in lines]
    print(f"--> # emails = {len(documents)}")

    return documents


def generate_synthetic_instructions(
    documents: List[Document], writer: PanzaWriter, batch_size: int, output_path: str
) -> None:
    num_processed_documents = 0
    num_batches = (len(documents) - 1) // batch_size + 1
    start_time = time.time()
    with open(output_path, "w") as f:
        for i in tqdm(range(0, len(documents), batch_size)):
            print(f"--> Processing batch {i // batch_size + 1}/{num_batches}")
            batch = documents[i : i + batch_size]
            instructions = [
                SummarizationInstruction(instruction=document.email) for document in batch
            ]

            summaries = writer.run_batch(instructions)
            num_processed_documents += len(summaries)

            for it, summary in enumerate(summaries):
                # Considerf adding cleaning and filtering here.
                batch[it].summary = summary

            # Write the summarized documents to a file
            for document in batch:
                f.write(json.dumps(document.serialize()))
                f.write("\n")

    elapsed_time = time.time() - start_time
    LOGGER.info(f"--> Processed {num_processed_documents} documents in {elapsed_time:.2f} seconds.")


def check_if_file_exists(cfg: DictConfig) -> None:
    if os.path.exists(cfg.cleaned_emails_path) and not cfg.force_extract_clean_emails:
        LOGGER.warning(
            f"Cleaned email file already exists, using existing file {cfg.cleaned_emails_path}. "
            "If you want to regenerate use the flag force_extract_clean_emails=true."
        )
        return True
    return False


def split_and_write_data(cfg):
    if cfg.test_split == 0:
        shutil.copy(cfg.summarized_emails_path, os.path.join(cfg.user.data_dir, "train.jsonl"))
        # Bad hack - we need test data for the training to work.
        shutil.copy(cfg.summarized_emails_path, os.path.join(cfg.user.data_dir, "test.jsonl"))
    else:
        with open(cfg.summarized_emails_path, "r") as f:
            data = f.readlines()
        if cfg.split_type == "random":
            random.seed(cfg.seed)
            random.shuffle(data)
        elif cfg.split_type == "chronological":
            data = sorted(data, key=lambda x: datetime.fromisoformat(json.loads(x)["date"]))
        else:
            raise ValueError("Invalid split type.")

        train_size = int(len(data) * (1 - cfg.test_split))

        with open(os.path.join(cfg.user.data_dir, "train.jsonl"), "w") as f:
            for i in range(train_size):
                f.write(data[i])

        with open(os.path.join(cfg.user.data_dir, "test.jsonl"), "w") as f:
            for i in range(train_size, len(data)):
                f.write(data[i])


@hydra.main(version_base="1.1", config_path="../configs", config_name="panza_preparation")
def main(cfg: DictConfig) -> None:
    LOGGER.info("Running Panza Data Preparation")
    LOGGER.info("Configuration: \n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Rename config keys to follow class structure
    rename_config_keys(cfg)

    # Skip running if  already exist
    if not check_if_file_exists(cfg):
        # Extract the emails from the .mbox file
        extract_emails(
            cfg.email_dump_path,
            cfg.cleaned_emails_path,
            [cfg.user.email_address],
            cfg.discarded_emails_dir,
        )

    # Instantiate Panza writer
    writer: PanzaWriter = hydra.utils.instantiate(cfg.writer)
    assert isinstance(writer, PanzaWriter), "Failed to instantiate PanzaWriter"

    # Load documents
    documents = load_documents(cfg.cleaned_emails_path)
    generate_synthetic_instructions(
        documents=documents,
        writer=writer,
        batch_size=cfg.batch_size,
        output_path=cfg.summarized_emails_path,
    )

    # Write the test data to test.jsonl, with an optional train-test split
    split_and_write_data(cfg)

    # Use only the training data (which might be all the data) for RAG.
    create_vector_store(
        os.path.join(cfg.user.data_dir, "train.jsonl"),
        cfg.rag_embedding_chunk_size,
        cfg.rag_embedding_chunk_overlap,
        cfg.rag_db_dir,
        cfg.user.username,
        cfg.rag_embedding_model,
    )


if __name__ == "__main__":
    main()
