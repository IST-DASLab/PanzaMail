#!/bin/bash

source config.sh

MODEL="../checkpoints/models/panza_seanyang711_llama3_bf16-bs8-rosa_wl16_d0.01_1grads_mean_squared_r8_loralr1e-5_alpha16-lr1e-5-epochs5-wu8-seed42-PREAMBLE-16296"
MODEL="../checkpoints/models/panza_llama3_bf16-bs8-rosa_wl16_d0_1grads_mean_squared_r8_loralr1e-5_alpha16-lr1e-5-epochs5-wu8-seed42-PREAMBLE-31921"
MODEL="../checkpoints/models/panza_jen.iofinova-Meta-Llama-3-8B-Instruct-bf16-bs8-fft-lr1e-05-3ep-seed41"

DEVICE="cuda:1"
DTYPE="auto"

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

USE_RAG=$([ "${PANZA_DISABLE_RAG_INFERENCE}" = "1" ] && echo "" || echo "--use-rag")
USE_4BIT_QUANT=$([ "${MODEL_PRECISION}" = "4bit" ] && echo "--load-in-4bit" || echo "")

INFERENCE_SCRIPT=${PANZA_WORKSPACE}/src/panza/evaluation/service_inference.py
python ${INFERENCE_SCRIPT} \
    --model=${MODEL} \
    --device=${DEVICE} \
    --dtype=${DTYPE} \
    --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
    --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
    --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
    --embedding-model=${PANZA_EMBEDDING_MODEL} \
    --db-path=${PANZA_DATA_DIR} \
    --index-name=${PANZA_USERNAME} \
    --rag-relevance-threshold=${PANZA_RAG_RELEVANCE_THRESHOLD} \
    ${USE_4BIT_QUANT}
