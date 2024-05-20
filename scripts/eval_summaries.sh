set -e

source config.sh



EVAL_SCRIPT=${PANZA_WORKSPACE}/src/panza/evaluation/evaluate_summaries.py

# Uncomment this one to run on a golden file
python ${EVAL_SCRIPT} \
 	--model=$PANZA_GENERATIVE_MODEL \
  --golden-loc=${PANZA_WORKSPACE}/data/david_ground_truth_summaries.jsonl \
   --prompt-file=/nfs/scistore19/alistgrp/eiofinov/PanzaMail/src/panza/data_preparation/summarization_prompt.txt \
   --batch-size 8


# Uncomment this one to run on the test file.
# python ${EVAL_SCRIPT} \
# 	--model=$PANZA_GENERATIVE_MODEL \
#   --summarized-emails-file=${PANZA_WORKSPACE}/data/test.jsonl \
#   --prompt-file=/nfs/scistore19/alistgrp/eiofinov/PanzaMail/src/panza/data_preparation/summarization_prompt.txt \
#   --batch-size 8
