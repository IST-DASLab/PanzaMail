#!/bin/bash

source config.sh

TRAIN_RATIO=1.0
SPLIT_TYPE="chronological"  # random or chronological

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

# Create synthetic instructions (summaries) for emails
python ../src/panza/data_preparation/summarize_emails.py \
    --path-to-emails="${PANZA_DATA_DIR}/${PANZA_USERNAME}_clean.jsonl" \
    --prompt-file="${PANZA_WORKSPACE}/src/panza/data_preparation/summarization_prompt.txt" \
    --batch-size=${PANZA_SUMMARIZATION_BATCH_SIZE} ${USE_4BIT_QUANT} ${USE_FP32_COMPUTE} &&

if [[ $TRAIN_RATIO < 1.0 ]]; then
    # Create train and test splits
    SPLIT_PANZA_DATA_DIR=${PANZA_DATA_DIR}/split

    python ../src/panza/data_preparation/split_data.py \
        --data-path="${PANZA_DATA_DIR}/${PANZA_USERNAME}_clean_summarized.jsonl" \
        --output-data-dir=${PANZA_DATA_DIR}/split \
        --train-ratio=${TRAIN_RATIO} \
        --split-type=${SPLIT_TYPE} \
        --seed=${PANZA_SEED}

    PANZA_DATA_DIR=$SPLIT_PANZA_DATA_DIR
else
    cp "${PANZA_DATA_DIR}/${PANZA_USERNAME}_clean_summarized.jsonl" \
        "${PANZA_DATA_DIR}/train.jsonl"

    # Finetuning requires some sort of test set, just use the training
    # data again.
    cp "${PANZA_DATA_DIR}/${PANZA_USERNAME}_clean_summarized.jsonl" \
        "${PANZA_DATA_DIR}/test.jsonl"
fi

# Create vector store with emails embeddings
# Note that if the data is split, then the PANZA_DATA_DIR,
# where the vector store will be, will be the /split directory.
python ../src/panza/data_preparation/create_vector_store.py \
    --path-to-emails="${PANZA_DATA_DIR}/train.jsonl" \
    --chunk-size=${CHUNK_SIZE} \
    --chunk-overlap=${CHUNK_OVERLAP} \
    --db-path=${PANZA_DATA_DIR} \
    --index-name=${PANZA_USERNAME} \
    --embedding_model=${PANZA_EMBEDDING_MODEL}
