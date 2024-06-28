#!/bin/bash

source config.sh

BASE_MODEL=${PANZA_GENERATIVE_MODEL}
MODEL=
DATA_PATH=${PANZA_DATA_DIR}/train.jsonl
NBITS=4

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

OUTPUT_PATH=${MODEL}/gptq-${NBITS}bits/
GTPQ_SCRIPT=${PANZA_WORKSPACE}/src/panza/finetuning/rosa_to_gptq.py
python ${GTPQ_SCRIPT} \
    --base-model=${BASE_MODEL} \
    --adapter=${MODEL} \
    --nbits=${NBITS} \
    --data-path=${DATA_PATH} \
    --output-path=${OUTPUT_PATH} \
    --num-samples=${GPTQ_NUM_SAMPLES} \
    --system-preamble=${PANZA_SYSTEM_PREAMBLE_PATH} \
    --user-preamble=${PANZA_USER_PREAMBLE_PATH}

echo "find the gptq model at ${OUTPUT_PATH}"