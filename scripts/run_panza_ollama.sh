#!/bin/bash

source config.sh

MODEL=${PANZA_GENERATIVE_MODEL}  # Replace this with the checkpoint you want to use!

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

USE_RAG=$([ "${PANZA_DISABLE_RAG_INFERENCE}" = "1" ] && echo "" || echo "--use-rag")
USE_4BIT_QUANT=$([ "${MODEL_PRECISION}" = "4bit" ] && echo "--load-in-4bit" || echo "")

INFERENCE_SCRIPT=${PANZA_WORKSPACE}/src/panza/evaluation/ollama_inference.py
python ${INFERENCE_SCRIPT} \
    --model=llama3.1 \
    --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
    --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
    --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
    --embedding-model=${PANZA_EMBEDDING_MODEL} \
    --db-path=${PANZA_DATA_DIR} \
    --index-name=${PANZA_USERNAME} \
    --rag-relevance-threshold=${PANZA_RAG_RELEVANCE_THRESHOLD} \
    ${USE_RAG} \
    ${USE_4BIT_QUANT}
