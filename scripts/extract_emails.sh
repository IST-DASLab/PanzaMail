#!/bin/bash

source config.sh

MBOX_NAME="Sent.mbox"
MBOX_PATH="${PANZA_DATA_DIR}/${MBOX_NAME}"

python ../src/panza/data_preparation/extract_emails.py \
    --mbox-path=${MBOX_PATH} \
    --output-path=${PANZA_DATA_DIR} \
    --email=${PANZA_EMAIL_ADDRESS} \