# Convenience script for combining all data preparation, model training
# and model evaluation with json
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# CUDA_VISIBLE_DEVICES=x ./prepare_train_eval.sh user=alonso finetuning=rosa

set -e

vars=()
idx=1

# process input arguments
training_mode="tbd" # training_mode to be determined later.
test_split="0"
for argument in "$@"
do
    key=$(echo $argument | cut -f1 -d=)
    if [[ $key == test_split ]]; then
        test_split=${argument#*=}
        echo "Setting the test_split to ${test_split}"
    elif [[ $key == finetuning ]]; then
        training_mode=${argument#*=}
        echo "Setting finetuning mode to ${training_mode}"
    elif [[ $training_mode == "rosa" ]] && [[ $key == finetuning.rosa.masks_only ]];then
        echo "The 'finetuning.rosa.masks_only' argument is already set and should not be overridden here; override is ignored."
    else
        vars[idx]=$argument
        idx+=1
    fi
done

# Step 1. Prepare the data
python ./prepare_data.py ${vars[@]}
# Step 2 & 3 Combined. Determine the type of training to do and evaluate with json.
if [[ $training_mode == "rosa" ]]; then
    # First create the masks for RoSA finetuning.
    composer ../src/panza/finetuning/train.py \
        finetuning=rosa finetuning.rosa.masks_only=true ${vars[@]}
    # Then train the weights.
    composer ../src/panza/finetuning/train.py \
        finetuning=rosa finetuning.rosa.masks_only=false ${vars[@]}
    if [[ $test_split != "0" ]]; then
        echo "Generating json evaluation"
        python runner.py interfaces=json writer/llm=peft 
    fi
elif [[ $training_mode == "full" ]];  then
    composer ../src/panza/finetuning/train.py \
        finetuning=full ${vars[@]}
    if [[ $test_split != "0" ]]; then
        echo "Generating json evaluation"
        python runner.py interfaces=json writer/llm=transformers 
    fi    
fi