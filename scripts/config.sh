#!/bin/bash

export PANZA_EMAIL_ADDRESS="armand.nicolicioiu@gmail.com"  # Change this to your email address!
export PANZA_USERNAME="${PANZA_EMAIL_ADDRESS%@*}"  # Removes everything after @

export PANZA_WORKSPACE=$(dirname "$(dirname "$(realpath "$0")")");
export PANZA_DATA_DIR="$PANZA_WORKSPACE/data"
export PANZA_CHECKPOINTS="$PANZA_WORKSPACE/checkpoints"
export PANZA_FINETUNE_CONFIGS="$PANZA_WORKSPACE/src/panza/finetuning/configs"

export PANZA_PREAMBLES="$PANZA_WORKSPACE/prompt_preambles"
export PANZA_SYSTEM_PREAMBLE_PATH="$PANZA_PREAMBLES/system_preamble.txt"
export PANZA_USER_PREAMBLE_PATH="$PANZA_PREAMBLES/user_preamble.txt"
export PANZA_RAG_PREAMBLE_PATH="$PANZA_PREAMBLES/rag_preamble.txt"

export PANZA_GENERATIVE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
export PANZA_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"

export PANZA_RAG_RELEVANCE_THRESHOLD=0.2

export PANZA_SEED=42

export PANZA_FINETUNE_WITH_PREAMBLE=1
export PANZA_DISABLE_RAG_INFERENCE=0

export PYTHONPATH="$PANZA_WORKSPACE/src:$PYTHONPATH"
