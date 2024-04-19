set -e

source config.sh

current_user=$(whoami)

export DATA_PATH=${PANZA_DATA_DIR}/train.jsonl

# hyper-parameters with default values
export MODEL_PRECISION=bf16 # bf16 or fp32
export BASE_SAVE_PATH=${PANZA_CHECKPOINTS} # where to store the model
export NUM_EPOCHS=3
export WARMUP=20 # the learning rate warmup (batches)
export BS=8
export PER_DEVICE_BS=1
export SEED=${PANZA_SEED}
export LR=1e-6 # learning rate

export WANDB_PROJECT="panza-${current_user}-test"

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
export PRETRAINED=${PANZA_GENERATIVE_MODEL}
export MAX_DURATION=${NUM_EPOCHS}ep
export CONFIG=${PANZA_FINETUNE_CONFIGS}/mistral_7b_fft_panza.yaml
export RUN_NAME=panza_${MODEL_PRECISION}-bs${BS}-fft-lr${LR}-epochs${NUM_EPOCHS}-wu${WARMUP}-seed${SEED}-${PREAMBLE_STR}-$RANDOM

# create directories to save the models
mkdir -p ${BASE_SAVE_PATH}/models/

TEMP_FILE=$(mktemp)

TRAIN_SCRIPT=${PANZA_WORKSPACE}/src/panza/finetuning/train.py
composer ${TRAIN_SCRIPT} \
    ${CONFIG} \
    model_name_or_path=${PRETRAINED} \
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
    global_seed=${SEED} \
    seed=${SEED} \
    hf_save_path=${BASE_SAVE_PATH}/models/ 2>&1 | tee "$TEMP_FILE"

# Extract the wandb run ID from the temp file
WANDB_RUN_ID=$(grep -o 'https://wandb.ai/[^ ]*/runs/[^ ]*' "$TEMP_FILE" | awk -F'/' '{print $NF}' | tail -n 1)

rm "$TEMP_FILE"

echo "find the finetuned model at ${BASE_SAVE_PATH}/models/${RUN_NAME}"

if [ -z "$WANDB_RUN_ID" ]; then
  echo "Failed to extract wandb run ID"
else
  echo "Extracted wandb run ID: $WANDB_RUN_ID"
  # Running BLEU evaluation
  EVAL_SCRIPT=${PANZA_WORKSPACE}/src/panza/evaluation/evaluate_bleu_score.py
  python ${EVAL_SCRIPT} \
  --model=${BASE_SAVE_PATH}/models/${RUN_NAME} \
  --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
  --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
  --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
  --golden=${PANZA_DATA_DIR}/test.jsonl \
  --wandb-run-id=${WANDB_RUN_ID}

  # Running BLEU evaluation with RAG
  python ${EVAL_SCRIPT} \
  --model=${BASE_SAVE_PATH}/models/${RUN_NAME} \
  --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
  --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
  --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
  --golden=${PANZA_DATA_DIR}/test.jsonl \
  --wandb-run-id=${WANDB_RUN_ID} \
  --embedding-model=${PANZA_EMBEDDING_MODEL} \
  --db-path=${PANZA_DATA_DIR} \
  --index-name=${PANZA_USERNAME} \
  --use-rag

  echo "find the finetuned model at ${BASE_SAVE_PATH}/models/${RUN_NAME}"
fi
