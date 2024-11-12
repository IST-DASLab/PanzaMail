set -e

source config.sh

current_user=$(whoami)

export DATA_PATH=${PANZA_DATA_DIR}/train.jsonl

# hyper-parameters with default values
export MASK_GEN_MODEL_PRECISION=${MODEL_PRECISION} # bf16, fp32, or 4bit
export BASE_SAVE_PATH=${PANZA_CHECKPOINTS} # where to store the checkpoints and generated masks
export NUM_EPOCHS=5
export WARMUP=8 # the learning rate warmup (batches)
export BS=8
export PER_DEVICE_BS=1
export LORA_ALPHA=16
export SCHEDULE=wl16 # the RoSA schedule
export SPA_NUM_GRADS=1 # number of gradients used for mask generation
export SPA_GRAD_ACC_MODE=mean_squared # 'mean' or 'mean_squared': how to accumulate gradients
export SEED=${PANZA_SEED}


export PANZA_RAG_RELEVANCE_THRESHOLD=0 # emails whose relevance is above this threshold will be presented for RAG 

if [[ ${MODEL_TYPE} == llama3 ]]; then 
    export LR=1e-5 # learning rate
    export LORA_LR=1e-5 # a separate learning rate for the low-rank adapters
elif [[ ${MODEL_TYPE} == mistralv2 ]]; then 
    export LR=1e-5 # learning rate
    export LORA_LR=1e-5 # a separate learning rate for the low-rank adapters
else
    echo "Model type ${MODEL_TYPE} not recognized! Panza only works with mistralv2 and llama3 models. Exiting."
    exit
fi

echo "Using Learning Rate ${LR} and LoRA LR ${LORA_LR} for ${MODEL_TYPE} model"


# hyper-parameters without default values
export SPA_DENSITY=0.01 # the sparse adapters' density
export LORA_R=8 # the low-rank adapters' rank

export WANDB_PROJECT="panza-${current_user}"
export PRETRAINED=${PANZA_GENERATIVE_MODEL}
export CONFIG=${PANZA_FINETUNE_CONFIGS}/rosa_panza.yaml
export NUM_CPU_THREADS=0 # useful for running of CPU, 0 means default the used by torch

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" # if not set, default to 0

# take all the input arguments and put them in environment variables
# this could override the hyper-parameters defined above
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

if [ "$PANZA_FINETUNE_WITH_PREAMBLE" = 1 ]; then
  PREAMBLE_STR="PREAMBLE"
  PREPROCESSING_FN=panza.finetuning.preprocessing:panza_preprocessing_function_train_with_preamble
else
  PREAMBLE_STR=""
  PREPROCESSING_FN=panza.finetuningpreprocessing:panza_preprocessing_function
fi

# some post-processing on the inputs




echo $RUN_NAME
# Running BLEU evaluation
EVAL_SCRIPT=${PANZA_WORKSPACE}/src/panza/evaluation/evaluation.py
# python ${EVAL_SCRIPT} \
#   --model=${BASE_SAVE_PATH}/models/${RUN_NAME} \
#   --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
#   --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
#   --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
#   --golden=${PANZA_DATA_DIR}/test.jsonl \
#   --batch-size=${PANZA_EVALUATION_BATCH_SIZE} \
#   --wandb-run-id=${WANDB_RUN_ID} \
#   ${USE_4BIT_QUANT}

  #--model=/nfs/scistore19/alistgrp/eiofinov/.cache/huggingface/hub/models--ISTA-DASLab--Meta-Llama-3-8B-Instruct/snapshots/0e6f530447ceec1aea4fd96e2aafad06bb3aa4b5/ \
# Running BLEU evaluation with RAG
python ${EVAL_SCRIPT} \
   --model=${BASE_SAVE_PATH}/models/${RUN_NAME} \
  --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
  --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
  --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
  --golden=${PANZA_DATA_DIR}/test.jsonl \
  --batch-size=${PANZA_EVALUATION_BATCH_SIZE} \
  --wandb-run-id=${WANDB_RUN_ID} \
  --embedding-model=${PANZA_EMBEDDING_MODEL} \
  --db-path=${PANZA_DATA_DIR} \
  --index-name=${PANZA_USERNAME} \
  --use-rag \
  ${USE_4BIT_QUANT}
