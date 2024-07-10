#!/bin/bash

source config.sh

TRAIN_RATIO=0.8

CHUNK_SIZE=3000
CHUNK_OVERLAP=3000

LOAD_IN_4BIT=0
RUN_FP32=0

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

USE_4BIT_QUANT=$([ "${LOAD_IN_4BIT}" = 1 ] && echo "--load-in-4bit" || echo "")
USE_FP32_COMPUTE=$([ "${RUN_FP32}" = 1 ] && echo "--fp32" || echo "")

# Create synthetic instructions (summaries) for the pieces of text
python ../src/panza/data_preparation/summarize_emails.py \
    --path-to-emails="${PANZA_DATA_DIR}/${PANZA_USERNAME}_clean.jsonl" \
    --prompt-file="${PANZA_WORKSPACE}/src/panza/data_preparation/summarization_prompt.txt" \
    --batch-size=${PANZA_SUMMARIZATION_BATCH_SIZE} ${USE_4BIT_QUANT} ${USE_FP32_COMPUTE} &&

# Create train and test splits
python ../src/panza/data_preparation/split_data.py \
    --data-path="${PANZA_DATA_DIR}/${PANZA_USERNAME}_clean_summarized.jsonl" \
    --output-data-dir=${PANZA_DATA_DIR} \
    --train-ratio=${TRAIN_RATIO} \
    --seed=${PANZA_SEED} &&

# Create vector store with text embeddings
python ../src/panza/data_preparation/create_vector_store.py \
    --path-to-emails="${PANZA_DATA_DIR}/train.jsonl" \
    --chunk-size=${CHUNK_SIZE} \
    --chunk-overlap=${CHUNK_OVERLAP} \
    --db-path=${PANZA_DATA_DIR} \
    --index-name=${PANZA_USERNAME} \
    --embedding_model=${PANZA_EMBEDDING_MODEL}
