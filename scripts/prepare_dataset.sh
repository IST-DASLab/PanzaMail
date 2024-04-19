#!/bin/bash

source config.sh

TRAIN_RATIO=0.8

CHUNK_SIZE=3000
CHUNK_OVERLAP=3000

# Create synthetic instructions (summaries) for emails
python ../src/panza/data_preparation/summarize_emails.py \
    --path-to-emails="${PANZA_DATA_DIR}/${PANZA_USERNAME}_clean.jsonl" \
    --prompt-file="${PANZA_WORKSPACE}/src/panza/data_preparation/summarization_prompt.txt" &&

# Create train and test splits
python ../src/panza/data_preparation/split_data.py \
    --data-path="${PANZA_DATA_DIR}/${PANZA_USERNAME}_clean_summarized.jsonl" \
    --output-data-dir=${PANZA_DATA_DIR} \
    --train-ratio=${TRAIN_RATIO} \
    --seed=${PANZA_SEED} &&

# Create vector store with emails embeddings
python ../src/panza/data_preparation/create_vector_store.py \
    --path-to-emails="${PANZA_DATA_DIR}/train.jsonl" \
    --chunk-size=${CHUNK_SIZE} \
    --chunk-overlap=${CHUNK_OVERLAP} \
    --db-path=${PANZA_DATA_DIR} \
    --index-name=${PANZA_USERNAME} \
    --embedding_model=${PANZA_EMBEDDING_MODEL}
