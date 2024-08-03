#!/bin/bash

export PANZA_EMAIL_ADDRESS="firstname.lastname@gmail.com"  # Change this to your email address!
export PANZA_USERNAME="${PANZA_EMAIL_ADDRESS%@*}"  # Removes everything after @; for the example above, it will be firstname.lastname

export PANZA_WORKSPACE=$(dirname "$(dirname "$(realpath "$0")")");
export PANZA_DATA_DIR="$PANZA_WORKSPACE/data"  # where data is stored
export PANZA_CHECKPOINTS="$PANZA_WORKSPACE/checkpoints" # where checkpoints are stored
export PANZA_FINETUNE_CONFIGS="$PANZA_WORKSPACE/src/panza/finetuning/configs" # where training configuration details are stored

export PANZA_PREAMBLES="$PANZA_WORKSPACE/prompt_preambles" # this is where the system prompt and user prompt preambles can be accessed; you will need to edit these
export PANZA_SYSTEM_PREAMBLE_PATH="$PANZA_PREAMBLES/system_preamble.txt"  # system prompt
# IMPORTANT: Please edit the user preamble (at the PANZA_USER_PREAMBLE_PATH) if you plan to use it (recommended).
export PANZA_USER_PREAMBLE_PATH="$PANZA_PREAMBLES/user_preamble.txt" # a useful preamble to the user instruction, explaining what's going on to the LLM
export PANZA_RAG_PREAMBLE_PATH="$PANZA_PREAMBLES/rag_preamble.txt"  # a preamble for the RAG component

export PANZA_SUMMARIZATION_BATCH_SIZE=8  # batch size for summarization.
export PANZA_EVALUATION_BATCH_SIZE=1  # batch size for evaluation. Can safely be set to higher value (e.g., 8) if the GPU has enough capacity.

export MODEL_PRECISION=bf16 # precision at which the base model is stored; options: bf16, fp32, or '4bit'
# export PANZA_GENERATIVE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
# export PANZA_GENERATIVE_MODEL="ISTA-DASLab/Meta-Llama-3-8B-Instruct"
export PANZA_GENERATIVE_MODEL="microsoft/Phi-3-mini-4k-instruct"
# export PANZA_GENERATIVE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

lowercased=$(echo "$PANZA_GENERATIVE_MODEL" | tr '[:upper:]' '[:lower:]')
if [[ ${lowercased} == *llama* ]]; then
    export MODEL_TYPE=llama3
elif [[ ${lowercased} == *mistral* ]]; then
    export MODEL_TYPE=mistralv2
elif [[ ${lowercased} == *phi* ]]; then
    export MODEL_TYPE=phi3
else
    echo "Model type ${PANZA_GENERATIVE_MODEL} not recognized! Panza only works with Mistral and Llama3 models. Exiting."
    exit
fi

export PANZA_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2" # embedding model for RAG; can be changed, trading off speed for quality

export PANZA_RAG_RELEVANCE_THRESHOLD=0.2 # emails whose relevance is above this threshold will be presented for RAG 

export PANZA_SEED=42 # the one true seed

export PANZA_FINETUNE_WITH_PREAMBLE=1  # states whether user and system preambles are used for fine-tuning; on by default
export PANZA_FINETUNE_WITH_RAG=0  # states whether RAG preambles are used for fine-tuning; on by default
export PANZA_FINETUNE_RAG_NUM_EMAILS=3  # maximum number of emails to use for RAG fine-tuning; 3 by default
export PANZA_FINETUNE_RAG_PROB=0.55  # probability of using RAG context for fine-tuning; 0.5 by default
export PANZA_FINETUNE_RAG_RELEVANCE_THRESHOLD=0.2  # emails whose relevance is above this threshold will be presented for RAG during fine-tuning
export PANZA_DISABLE_RAG_INFERENCE=0  # RAG inference is on by default, since it's usually better

export PANZA_WANDB_DISABLED=True  # disable Weights and Biases logging by default

export PYTHONPATH="$PANZA_WORKSPACE/src:$PYTHONPATH"

# Optionally, set your HF_HOME and/or TRANSFORMERS_CACHE here.
# export HF_HOME=
# export TRANSFORMERS_CACHE=
