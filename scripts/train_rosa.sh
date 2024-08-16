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

if [[ ${MODEL_TYPE} == llama3 ]]; then
    export LR=1e-5 # learning rate
    export LORA_LR=1e-5 # a separate learning rate for the low-rank adapters
elif [[ ${MODEL_TYPE} == mistralv2 ]]; then
    export LR=1e-5 # learning rate
    export LORA_LR=1e-5 # a separate learning rate for the low-rank adapters
elif [[ ${MODEL_TYPE} == phi3 ]]; then
    export LR=1e-5 # learning rate
    export LORA_LR=1e-5 # a separate learning rate for the low-rank adapters
else
    echo "Model type ${MODEL_TYPE} not recognized! Panza only works with mistralv2, llama3 and phi3 models. Exiting."
    exit
fi

# hyper-parameters without default values
export SPA_DENSITY=0.01 # the sparse adapters' density
export LORA_R=8 # the low-rank adapters' rank

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

echo "Using Learning Rate ${LR} and LoRA LR ${LORA_LR} for ${MODEL_TYPE} model"

export WANDB_PROJECT="panza-${PANZA_USERNAME}"

if [ "$PANZA_FINETUNE_WITH_PREAMBLE" = 1 ]; then
  PREAMBLE_STR="-PREAMBLE"
  PREPROCESSING_FN=panza.finetuning.preprocessing:panza_preprocessing_function_train_with_preamble
else
  PREAMBLE_STR=""
  PREPROCESSING_FN=panza.finetuning.preprocessing:panza_preprocessing_function
fi

if [ "$PANZA_FINETUNE_WITH_RAG" = 1 ]; then
  RAFT_STR=-RAFT_num${PANZA_FINETUNE_RAG_NUM_EMAILS}_prob${PANZA_FINETUNE_RAG_PROB}_th${PANZA_FINETUNE_RAG_RELEVANCE_THRESHOLD}
else
  RAFT_STR=""
fi

# some post-processing on the inputs
export MAX_DURATION=${NUM_EPOCHS}ep
export RUN_NAME=panza_${PANZA_USERNAME}_${MODEL_TYPE}_${MODEL_PRECISION}-bs${BS}-rosa_${SCHEDULE}_d${SPA_DENSITY}_${SPA_NUM_GRADS}grads_${SPA_GRAD_ACC_MODE}_r${LORA_R}_loralr${LORA_LR}_alpha${LORA_ALPHA}-lr${LR}-epochs${NUM_EPOCHS}-wu${WARMUP}-seed${SEED}${PREAMBLE_STR}${RAFT_STR}-$RANDOM

# create directories to save the masks and models
mkdir -p ${BASE_SAVE_PATH}/masks/
mkdir -p ${BASE_SAVE_PATH}/models/

if [ "$MODEL_PRECISION" = "bf16" ]; then
  export ROSA_DTYPE=bf16
elif [ "$MODEL_PRECISION" = "4bit" ]; then
  export ROSA_DTYPE=fp32
elif [ "$MODEL_PRECISION" = "fp32" ]; then
  export ROSA_DTYPE=fp32
else
  echo "Unknown model precision $MODEL_PRECISION"
  exit 1
fi

if [[ "$SPA_DENSITY" != "0" ]]
then
    # sparse adaptation exists, so we need to generate masks

    if [[ $LORA_R == 0 ]]
    then
        export SCHEDULE=spa_only
    fi

    # no wandb logging for mask generation
    export WANDB_DISABLED=True

    # generate the masks and terminate
    TRAIN_SCRIPT=${PANZA_WORKSPACE}/src/panza/finetuning/train.py
    composer ${TRAIN_SCRIPT} \
        ${CONFIG} \
        model_name_or_path=${PRETRAINED} \
        num_cpu_threads=${NUM_CPU_THREADS} \
        train_loader.dataset.hf_kwargs.data_files=${DATA_PATH} \
        train_loader.dataset.preprocessing_fn=${PREPROCESSING_FN} \
        max_duration=${MAX_DURATION} \
        run_name=${RUN_NAME} \
        optimizer.lr=${LR} \
        global_train_batch_size=${BS} \
        device_train_microbatch_size=${PER_DEVICE_BS} \
        device_eval_batch_size=${PER_DEVICE_BS} \
        scheduler.t_warmup=${WARMUP}ba \
        model.weight_bias_dtype=${MASK_GEN_MODEL_PRECISION} \
        rosa.spa_d=${SPA_DENSITY} \
        rosa.spa_num_grads=${SPA_NUM_GRADS} \
        rosa.grad_acc_mode=${SPA_GRAD_ACC_MODE} \
        rosa.lora_r=${LORA_R} \
        rosa.lora_alpha=${LORA_ALPHA} \
        rosa.lora_lr=${LORA_LR} \
        rosa.schedule=${SCHEDULE} \
        rosa.rosa_dtype=${ROSA_DTYPE} \
        global_seed=${SEED} \
        seed=${SEED} \
        hf_save_path=${BASE_SAVE_PATH}/models/ \
        rosa.mask_save_path=${BASE_SAVE_PATH}/masks/${RUN_NAME} \
        rosa.terminate_after_mask_generation=true
fi

# now we have the masks ready, so let's restart
export MASK_LOAD_PATH=${BASE_SAVE_PATH}/masks/${RUN_NAME}

# determine the correct RoSA schedule
if [[ "$SPA_DENSITY" != "0" && $LORA_R -ne 0 ]]
then
    export SCHEDULE=default
elif [[ $LORA_R -ne 0 ]]
then
    export SCHEDULE=lora_only
    export MASK_LOAD_PATH=
else
    export SCHEDULE=spa_only
fi

TEMP_FILE=$(mktemp)

export WANDB_DISABLED=${PANZA_WANDB_DISABLED}
# start the training with both sparse and low-rank adapters active from the outset
TRAIN_SCRIPT=${PANZA_WORKSPACE}/src/panza/finetuning/train.py
composer ${TRAIN_SCRIPT} \
    ${CONFIG} \
    model_name_or_path=${PRETRAINED} \
    num_cpu_threads=${NUM_CPU_THREADS} \
    train_loader.dataset.hf_kwargs.data_files=${DATA_PATH} \
    train_loader.dataset.preprocessing_fn=${PREPROCESSING_FN} \
    max_duration=${MAX_DURATION} \
    run_name=${RUN_NAME} \
    optimizer.lr=${LR} \
    global_train_batch_size=${BS} \
    device_train_microbatch_size=${PER_DEVICE_BS} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    scheduler.t_warmup=${WARMUP}ba \
    model.weight_bias_dtype=${MODEL_PRECISION} \
    rosa.spa_d=${SPA_DENSITY} \
    rosa.spa_num_grads=${SPA_NUM_GRADS} \
    rosa.grad_acc_mode=${SPA_GRAD_ACC_MODE} \
    rosa.lora_r=${LORA_R} \
    rosa.lora_alpha=${LORA_ALPHA} \
    rosa.lora_lr=${LORA_LR} \
    rosa.schedule=${SCHEDULE} \
    rosa.rosa_dtype=${ROSA_DTYPE} \
    global_seed=${SEED} \
    seed=${SEED} \
    hf_save_path=${BASE_SAVE_PATH}/models/ \
    rosa.mask_load_path=${MASK_LOAD_PATH} 2>&1 | tee "$TEMP_FILE"

# Extract the wandb run ID from the temp file
WANDB_RUN_ID=$(grep -o 'https://wandb.ai/[^ ]*/runs/[^ ]*' "$TEMP_FILE" | awk -F'/' '{print $NF}' | tail -n 1)

# Clean up
rm "$TEMP_FILE"
rm -rf "$MASK_LOAD_PATH"

echo "find the adapter at ${BASE_SAVE_PATH}/models/${RUN_NAME}"

USE_4BIT_QUANT=$([ "${MODEL_PRECISION}" = "4bit" ] && echo "--load-in-4bit" || echo "")

if [ -z "$WANDB_RUN_ID" ]; then
  echo "No wandb run ID found."
else
  echo "Extracted wandb run ID: $WANDB_RUN_ID"
fi

# Running BLEU evaluation
EVAL_SCRIPT=${PANZA_WORKSPACE}/src/panza/evaluation/evaluation.py
python ${EVAL_SCRIPT} \
  --model=${BASE_SAVE_PATH}/models/${RUN_NAME} \
  --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
  --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
  --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
  --thread-preamble=${PANZA_THREAD_PREAMBLE_PATH} \
  --golden=${PANZA_DATA_DIR}/test.jsonl \
  --batch-size=${PANZA_EVALUATION_BATCH_SIZE} \
  --wandb-run-id=${WANDB_RUN_ID} \
  ${USE_4BIT_QUANT}

# Running BLEU evaluation with thread
EVAL_SCRIPT=${PANZA_WORKSPACE}/src/panza/evaluation/evaluation.py
python ${EVAL_SCRIPT} \
  --model=${BASE_SAVE_PATH}/models/${RUN_NAME} \
  --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
  --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
  --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
  --thread-preamble=${PANZA_THREAD_PREAMBLE_PATH} \
  --golden=${PANZA_DATA_DIR}/test.jsonl \
  --batch-size=${PANZA_EVALUATION_BATCH_SIZE} \
  --wandb-run-id=${WANDB_RUN_ID} \
  --use-thread \
  ${USE_4BIT_QUANT}

# Running BLEU evaluation with RAG
python ${EVAL_SCRIPT} \
  --model=${BASE_SAVE_PATH}/models/${RUN_NAME} \
  --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
  --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
  --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
  --thread-preamble=${PANZA_THREAD_PREAMBLE_PATH} \
  --golden=${PANZA_DATA_DIR}/test.jsonl \
  --batch-size=${PANZA_EVALUATION_BATCH_SIZE} \
  --wandb-run-id=${WANDB_RUN_ID} \
  --embedding-model=${PANZA_EMBEDDING_MODEL} \
  --db-path=${PANZA_DATA_DIR} \
  --index-name=${PANZA_USERNAME} \
  --use-rag \
  ${USE_4BIT_QUANT}

echo "find the adapter at ${BASE_SAVE_PATH}/models/${RUN_NAME}"
