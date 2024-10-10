set -e

source config.sh

# hyper-parameters with default values
#export MODEL_PRECISION=bf16 # bf16 or fp32
export BASE_SAVE_PATH=${PANZA_CHECKPOINTS} # where to store the model

TYPE=prompt
# take all the input arguments and put them in environment variables
# this could override the hyper-parameters defined above
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

if [[ $RUN_NAME == *"panza"* ]]; then
    MODEL=${BASE_SAVE_PATH}/models/${RUN_NAME}
else
    MODEL=$PANZA_GENERATIVE_MODEL
fi

EVAL_SCRIPT=${PANZA_WORKSPACE}/src/panza/evaluation/generate_questions.py

OUT_FOLDER=${PANZA_WORKSPACE}/human_eval
mkdir -p ${OUT_FOLDER}

if [[ $TYPE == prompt ]]; then

  file_arg="--prompts-file=${PANZA_WORKSPACE}/src/panza/evaluation/fixed_prompts.txt"

elif [[ $TYPE == own-data ]]; then
  file_arg="--test-data-file=${PANZA_WORKSPACE}/data/test.jsonl"

fi

python ${EVAL_SCRIPT} \
  --model=$MODEL \
  ${file_arg} \
  --out-path=${OUT_FOLDER} \
  --batch-size=${PANZA_EVALUATION_BATCH_SIZE} \
  --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
  --user-preamble=${PANZA_USER_PREAMBLE_PATH} \
  --rag-preamble=${PANZA_RAG_PREAMBLE_PATH} \
  --thread-preamble=${PANZA_THREAD_PREAMBLE_PATH} \
  --embedding-model=${PANZA_EMBEDDING_MODEL} \
  --db-path=${PANZA_DATA_DIR} \
  --index-name=${PANZA_USERNAME} \
  --use-rag

echo "find the evaluation file in ${OUT_FOLDER}"
